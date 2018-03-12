import os
import argparse
import time
import re
import pandas as pd

def is_valid_file(parser, arg):
  if not os.path.exists(arg):
    parser.error("The directory %s does not exist!" % arg)
  else:
    return arg

def create_dir_if_not_exist(parser, arg):
  if not os.path.exists(arg):
    os.makedirs(arg)
  else:
    return arg

def any_time_to_seconds(time_str):
  unit = str( re.sub("[^A-Za-z]", "", time_str) )
  unit = str( re.sub("e", "", unit ) )
  time_val = float( re.sub(unit, "", time_str) )

  if unit == 'ns':
    time_val /= (10.0 ** 9)
  elif unit == 'us':
    time_val /= (10.0 ** 6)
  elif unit == 'ms':
    time_val /= (10.0 ** 3)
  elif unit == 'm':
    time_val *= (60.0)
  elif unit == 'h':
    time_val *= (3600.0)

  return float(time_val)

def digits_items_in_list(List):
  filtered_list = []
  for element in List:
    if len(re.findall('\d+', element)) :
      filtered_list.append(element)
  return filtered_list

def read_and_process_file(input_file_name, input_dir, output_dir):
  if os.path.splitext(input_file_name)[1] != '.log':
    return -1

  (method, n_GPUs, n_obs, n_dim, K) = input_file_name.split('.')[0].split('-')
  n_obs = int ( n_obs.split('n_obs')[1] )
  n_dim = int ( n_dim.split('n_dims')[1] )
  n_GPUs = int ( n_GPUs.split('GPUs')[1] )
  K = int ( K.split('K')[1] )

  file_connection = open(input_dir + input_file_name, 'r')
  file_text = file_connection.read()
  file_connection.close()

  profling_calls_flag = '==\d+== Profiling result:'
  API_calls_flag = '==\d+== API calls:'
  
  try:
    after_profling_text = re.split(profling_calls_flag, file_text)[1]
    until_API_calls_text = re.split(API_calls_flag, after_profling_text)
  except IndexError:
    return -2

  profling_text = until_API_calls_text[0]
  API_calls_text = until_API_calls_text[1]

  profling_text_list = re.split('\n', profling_text)
  API_calls_text_list = re.split('\n', API_calls_text)

  ### Character Count
  profling_items_list = digits_items_in_list(profling_text_list)
  API_calls_items_list = digits_items_in_list(API_calls_text_list)

  line_split = []
  time_perc = []
  total_time = []
  n_calls = []
  avg_call_time = []
  min_call_time = []
  max_call_time = []
  call_name = []

  for item in profling_items_list:
    line_split = item.split()
    time_perc.append( line_split[0] )
    total_time.append( any_time_to_seconds(line_split[1]) )
    n_calls.append( line_split[2] )
    avg_call_time.append( any_time_to_seconds(line_split[3]) )
    min_call_time.append( any_time_to_seconds(line_split[4]) )
    max_call_time.append( any_time_to_seconds(line_split[5]) )
    call_name.append( ''.join(line_split[6:]) )
  aux_dict = {'TimePerc'    : time_perc    ,
              'Time'        : total_time   ,
              'NumCalls'    : n_calls      ,
              'AvgCallTime' : avg_call_time,
              'MinCallTime' : min_call_time,
              'MaxCallTime' : max_call_time,
              'CallName'    : call_name     }

  profling_result_data_frame = pd.DataFrame(aux_dict)
  save_file_name = os.path.join(output_dir, 'profling_result_' + os.path.splitext(input_file_name)[0] + '.csv')
  profling_result_data_frame.to_csv(save_file_name)

  line_split = []
  time_perc = []
  total_time = []
  n_calls = []
  avg_call_time = []
  min_call_time = []
  max_call_time = []
  call_name = []

  for item in API_calls_items_list:
    line_split = item.split()
    time_perc.append( line_split[0] )
    total_time.append( any_time_to_seconds(line_split[1]) )
    n_calls.append( line_split[2] )
    avg_call_time.append( any_time_to_seconds(line_split[3]) )
    min_call_time.append( any_time_to_seconds(line_split[4]) )
    max_call_time.append( any_time_to_seconds(line_split[5]) )
    call_name.append( ''.join(line_split[6:]) )
  aux_dict = {'TimePerc'    : time_perc    ,
              'Time'        : total_time   ,
              'NumCalls'    : n_calls      ,
              'AvgCallTime' : avg_call_time,
              'MinCallTime' : min_call_time,
              'MaxCallTime' : max_call_time,
              'CallName'    : call_name     }

  API_calls_data_frame = pd.DataFrame(aux_dict)
  save_file_name = os.path.join(output_dir, 
                  'API_calls_' + os.path.splitext(input_file_name)[0] + '.csv')
  API_calls_data_frame.to_csv(save_file_name)
  return 1


def main(input_dir, output_dir):
  all_input_files_input_dir = os.listdir(input_dir)
  
  I = 0
  for input_file_name in all_input_files_input_dir:
    I += 1
    print(I, '- input_file_name = ', input_file_name)
    read_and_process_file(input_file_name = input_file_name, 
                          input_dir = input_dir,
                          output_dir = output_dir)

  return 1

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description = 'Compile Distribuited K Means Results.')
  
  parser.add_argument("--input_dir"                                      ,
            dest = "input_dir"                                 ,
            required = True                                    ,
                        metavar = "FILE"                                   ,
                        type = lambda x: is_valid_file(parser, x)          ,
                        help = "The directory where log files are located!" )

  parser.add_argument('--output_dir'                                            ,
            dest = 'output_dir'                                       ,
            required = True                                           ,
                        metavar = "FILE"                                          ,
            type = lambda x: is_valid_file(create_dir_if_not_exist, x),
            help = 'The directory where log files will be saved!'      )

  args = parser.parse_args()
  input_dir = args.input_dir
  output_dir = args.output_dir

  main(input_dir = input_dir  ,
     output_dir = output_dir )