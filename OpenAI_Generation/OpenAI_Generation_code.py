import shutil
import time
from typing import final
from openai import  AsyncOpenAI
import json
import os
import asyncio
import pandas as pd

# OPENAI - Request ---------------------------------------------
async def OpenAI_QA_FunctionCall_general(message_content:str, model_name: str, openai_api, 
                                         model_tempreature:float=0, model_max_tokens:int=512,
                                         request_timeout:int=30,
                                         use_seed: bool=False):
    """
    Answer a medical question using the OPENAI language model asynchronously.

    Args:
        question (str): The medical question.
        choices (str): A string containing answer choices.

    Returns:
        tuple: A tuple containing the best answer, certainty, and rationale.
               If an error occurs, all values will be None.
               
    Notes: 2024027version: The token counter was added and returned. The timeout was added. The function will print the model parameters and prompts on the first run.
    """
    Experiment_detail={}
    Experiment_detail['overall_prompt'] = message_content
    Experiment_detail['model_temperature'] = model_tempreature 
    Experiment_detail['model_max_tokens'] = model_max_tokens
    Experiment_detail['seed']= 123 if use_seed else 'None'

    
    if not hasattr(OpenAI_QA_FunctionCall_general, 'has_run_before') or not OpenAI_QA_FunctionCall_general.has_run_before:
        print(f"This is the first time you run this function. The current parameters are: {str(Experiment_detail)}")
        OpenAI_QA_FunctionCall_general.has_run_before = True

    try:
    
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "structuring_output",
                    "description": "Predict the prognosis of a patient admitted with COVID-19.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "Outcome": {"type": "string",  
                                                "description": "The outcome of the patient during the admission. Possible values are 'survive' and 'die'.",
                                                "enum": ["die", "survive"]},
                        },
                        "required": ["Outcome"],
                    },
                },
            }
        ]
        
        messages=[
            {"role": "user", 
             "content": f"""
             {message_content}
             """  }]
        
        client = AsyncOpenAI(api_key=openai_api)

        if use_seed:
            seed_value = 123
        else:
            seed_value = None
            
        start_time = time.time()
        response  = await client.chat.completions.create(
            model=model_name,
            messages=messages,  
            max_tokens=model_max_tokens,
            temperature=model_tempreature,
            tools= tools,
            logprobs=False,
            timeout=request_timeout, 
            tool_choice= {"type": "function", "function": {"name": "structuring_output"}},
            seed=seed_value
            )
        end_time = time.time()
        execution_time = end_time - start_time
        Experiment_detail["execution_time"]=execution_time
               
        try:
            correct_answer_dic=json.loads(response.choices[0].message.tool_calls[0].function.arguments)
            structure_answer=correct_answer_dic["Outcome"]
        except Exception as ee:
            structure_answer=f'ERROR in extracting selected answer: {ee}'
    except Exception as e:
        Experiment_detail=response=structure_answer=f'ERROR in response generation: {e}'

    return Experiment_detail, response, structure_answer

def check_answer_correctness(answer, truth, reporterror_index=None,reporterror_name=None):
    if reporterror_index is None:
        reporterror_index=r'(IDK the index)'
    if reporterror_name is None:
        reporterror_name =r'(IDK the llm name)'
        
    norm_answer = str(answer).strip().upper() if answer else ''
    norm_truth = str(truth).strip().upper() if truth else ''


    # Check correctness
    if 'ERROR' in answer:
        # for cases that extraction of answer caused error
        return 'ERROR'
    
    if norm_answer and norm_truth:
        return 'correct' if norm_answer == norm_truth else 'incorrect'
    else:
        return 'ERROR: answer or truth invalid'

def save_and_open_excel(df, excel_output_path, open_at_end):
    try:
        df.to_excel(excel_output_path, index=False)
        print(f'Saved the excel file at {excel_output_path}')
    except Exception as e:
        print(f"Error saving the file to excel: {e}")
        return df

    if os.path.exists(excel_output_path) and open_at_end is True:
        try:
            os.startfile(excel_output_path)
        except Exception as e:
            print(f"Error opening excel: {e}")
            return df

    return df


async def handle_llm_response_functioncall_general(excel_file_path: str,  messsage_content_column:str, llm_list: list, openai_api, 
                                                   ground_truth_column:str = None,
                                                    open_at_end:bool=False,
                                                    number_of_task_to_save:int=15, add_delay_sec:int=1,
                                                    request_timeout:int=15,
                                                    model_tempreature=0, model_max_tokens=512, max_token_plus_input=False,
                                                    use_seed: bool=False
                                                    ):
    

    
    try:
        
        # Read excel
        df = pd.read_excel(excel_file_path,)
        
        async def process_row(row, llm_name, idx, model_max_tokens, max_token_plus_input,):
            if max_token_plus_input:
                input_token_count= row['input_token_count']
                
                max_token=input_token_count+model_max_tokens
                max_token=int(max_token)
            else:
                max_token=int(model_max_tokens)
                
            Experiment_detail= response=structure_answer= correctness = None
            Experiment_detail, response, structure_answer = await OpenAI_QA_FunctionCall_general(message_content=row[messsage_content_column], model_name=llm_name, openai_api=openai_api,request_timeout=request_timeout, 
                                                                                                model_tempreature=model_tempreature, model_max_tokens=max_token, 
                                                                                                    use_seed=use_seed)
            correctness = check_answer_correctness(truth=row[ground_truth_column], answer=structure_answer)
            return idx, Experiment_detail, response, structure_answer, correctness
    
        # Loop through llm list
        for llm in llm_list:
            # Create a column for each llm (to store response) if it doesn't exist
            if llm not in df.columns:
                df[llm] = ''
            
            tasks = []
            
            i=0
            max_index = df.index.max()
            for index, row in df.iterrows():
                if row[llm] != 'EXTRACTED':
                    

                    task = asyncio.create_task(process_row(row, llm, idx=index,model_max_tokens=model_max_tokens,max_token_plus_input=max_token_plus_input))
                    tasks.append(task)
                    i+=1
                    
                    #saving output after finishing number_of_task_to_save tasks    
                    if i==number_of_task_to_save or index == max_index: 
                        results = await asyncio.gather(*tasks)
                        
                        for result in results:
                            if result:  # Ensure result is not None or handle as needed

                                idx, Experiment_detail, response, structure_answer, correctness = result
                                # Update the DataFrame based on the result
                                df.at[idx, llm] = 'EXTRACTED'
                                df.at[idx, f'{llm}_rawresponse'] = str(response) 
                                df.at[idx, f'{llm}_outcome'] = str(structure_answer)
                                df.at[idx, f'{llm}_Experiment_detail'] = str(Experiment_detail)
                                df.at[idx, f'{llm}_correctness'] = correctness
                                
                                print(f"idx: {idx}, outcome: {structure_answer}")
                                
                        #save draft
                        try:
                            df.to_excel(excel_file_path)
                            print(f"Draft excel file saved at {excel_file_path}")
                        except Exception as e:
                            print(f"Error in saving temporary excel. Error:   {e}")
                            continue    
                            
                        
                        #reset for continue
                        i=0
                        tasks = []
                        print('sleep like a baby')
                        await asyncio.sleep(add_delay_sec)
            
    
    except KeyboardInterrupt or asyncio.CancelledError:
        print("Operation interrupted. Cleaning up...")
    
    except Exception as e:
        print(f"Error occured in the handler: {e}")
        
    finally:
        df = save_and_open_excel(df, excel_file_path, open_at_end)
        
        
        # reset the model experiemnt 
        OpenAI_QA_FunctionCall_general.has_run_before = False
        return df
        


excel_file_path=r"C:\Users\LEGION\Desktop\Code4ghaffar\zero_shot_by_hand.xlsx"
messsage_content_column="patient medical history"
llm_list=[
    "gpt-4o-2024-05-13", #	Up to Oct 2023   --> Gpt-4o
    "gpt-4-turbo-2024-04-09", #	Up to Dec 2023 --> Gpt-4turbo
    "gpt-4-0613", #	Up to Sep 2021 --> Gpt-4
    "gpt-3.5-turbo-0125" #	Up to September 2021 --> Gpt-3.5 
    ]
openai_api=os.getenv("OPENAI_API_KEY")
ground_truth_column="Inhospital Mortalit(TRUE)"
open_at_end=True
number_of_task_to_save=10
add_delay_sec=5
request_timeout=30

model_tempreature=1
model_max_tokens=1024
max_token_plus_input=False
use_seed=True


final_df = await handle_llm_response_functioncall_general(excel_file_path=excel_file_path,  messsage_content_column=messsage_content_column, llm_list=llm_list, openai_api=openai_api, 
                                                    ground_truth_column=ground_truth_column,
                                                    open_at_end=open_at_end,
                                                    number_of_task_to_save=number_of_task_to_save, add_delay_sec=add_delay_sec,
                                                    request_timeout=request_timeout,
                                                    
                                                    model_tempreature=model_tempreature, model_max_tokens=model_max_tokens, max_token_plus_input=max_token_plus_input,
                                                    use_seed=use_seed,
                                                    )

