#!/bin/bash

proyect_path="/media/Data/Shared/Documentos/Universidad/4º/Minería de Datos/Proyecto2"

src_folder="src"
main_file="main.py"
d2v_folder="d2v_models"
files_folder="files"
clean_data="verbal_autopsies_clean.csv"
results_folder="results"
results_sub_folder="naive_bayes"

cd "$proyect_path"

arg_data_path="-d \"$proyect_path/$files_folder/$clean_data\""
arg_output_path="-o \"$proyect_path/$results_folder/$results_sub_folder\""
arg_text_attr="-a open_response"
arg_class_attr="-c gs_text34"
arg_attr_conv="-d2v 50 2 40 \"$proyect_path/$d2v_folder/d2v_50_2_40.model\""
arg_classifier="-nb"
arg_k="-k 10"

command="python3 $src_folder/$main_file $arg_data_path $arg_output_path $arg_text_attr \
$arg_class_attr $arg_attr_conv $arg_classifier $arg_k"

echo -e "\033[32m$command\033[0m"
echo
eval $command
