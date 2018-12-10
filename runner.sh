#!/bin/bash

proyect_path="/media/Data/Shared/Documentos/Universidad/4º/Minería de Datos/Proyecto2"

src_folder="src"
main_file="main.py"
d2v_folder="d2v_models"
files_folder="files"
clean_data="verbal_autopsies_clean.csv"
results_folder="results"
# results_sub_folder="naive_bayes"
results_sub_folder="neural_network"

cd "$proyect_path"

arg_data_path="-d \"$proyect_path/$files_folder/$clean_data\""
# arg_output_path="-o \"$proyect_path/$results_folder/$results_sub_folder\""
arg_text_attr="-a open_response"
arg_class_attr="-c gs_text34"
arg_attr_conv="-d2v 50 2 40 \"$proyect_path/$d2v_folder/d2v_50_2_40.model\""
#arg_attr_conv="-t"
# arg_classifier="-nn 5 100"
#arg_classifier="-nb"
arg_k="-k 10"
arg_verbose="-v"


num_classifiers=( 2 3 4 5 6 7 8 9 10 )
num_layes=( 1 2 3 4 5 )
num_neurons=( 50 100 150 200 )

for c in "${num_classifiers[@]}"
do
    for l in "${num_layes[@]}"
    do
        for n in "${num_neurons[@]}"
        do
            neurons=""
            for ((i=1;i<=$l;i++))
            do
                neurons="$neurons$n "
            done
            arg_classifier="-nn $c $neurons"
            arg_output_path="-o \"$proyect_path/$results_folder/${results_sub_folder}_${c}_${l}_$n\""

            command="python3 $src_folder/$main_file $arg_data_path $arg_output_path $arg_text_attr \
            $arg_class_attr $arg_attr_conv $arg_classifier $arg_k $arg_verbose"

            echo -e "\033[32m$command\033[0m"
            echo
            eval $command
        done
    done
done


systemctl suspend
