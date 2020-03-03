#!/bin/bash

: '
/**

* © Copyright (C) 2016-2020 Xilinx, Inc
*
* Licensed under the Apache License, Version 2.0 (the "License"). You may
* not use this file except in compliance with the License. A copy of the
* License is located at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
* License for the specific language governing permissions and limitations
* under the License.
*/
'

# daniele.bagni@xilinx.com
# date 28 Feb 2020

## clean up previous log files
#rm -f ./log/*.log

CNN=unet

# folders
WORK_DIR=./workspace
LOG_DIR=${WORK_DIR}/../log
RPT_DIR=${WORK_DIR}/../rpt
TARGET_DIR=${WORK_DIR}/../target_zcu102/${CNN}
KERAS_MODEL_DIR=${WORK_DIR}/../keras_model
DCF_DIR=${WORK_DIR}/../dcf
DATASET_DIR=${WORK_DIR}/dataset1

TB_LOG_DIR=${WORK_DIR}/tb_log
CHKPT_DIR=${WORK_DIR}/tf_chkpts
FREEZE_DIR=${WORK_DIR}/freeze
COMPILE_DIR=${WORK_DIR}/compile
QUANT_DIR=${WORK_DIR}/quantize_results

# checkpoints & graphs filenames
CHKPT_FILENAME=float_model.ckpt
INFER_GRAPH_FILENAME=infer_graph.pb
FROZEN_GRAPH_FILENAME=frozen_graph.pb
QUANTIZED_FILENAME=quantize_eval_model.pb

# logs & results files
PREPARE_DATA_LOG=${CNN}_prepare_data.log
TRAIN_LOG=${CNN}_training.log
FREEZE_LOG=${CNN}_freeze_graph.log
EVAL_FR_LOG=${CNN}_evaluate_frozen_graph.log
QUANT_LOG=${CNN}_quantize.log
EVAL_Q_LOG=${CNN}_evaluate_quantized_graph.log
COMP_LOG=${CNN}_compile.log

# CNN parameters
INPUT_NODE="input_1"
OUTPUT_NODE="conv2d_19/Relu" # output node of floating point CNN UNET v1 and v3


# target DCF file
# taken from DNNDK3.1 installation
DCF_FILE=${DCF_DIR}/ZCU102.dcf
#DCF_FILE=${DCF_DIR}/ZCU104.dcf
#DCF_FILE=${DCF_DIR}/Ultra96.dcf
#DCF_FILE=${DCF_DIR}/ZedBoard.dcf

#use the patched DNNC version of ml1636_dnnc_package_20191017.tar.gz
#DNNC=${HOME}/ML/DNNDK3v1/patches/dnnc_20191017
DNNC=/opt/vitis_ai/compiler/dnnc/dpuv2/dnnc


##################################################################################
#setup the environment and check DNNDK relese
#source ${HOME}/scripts/activate_py36_dnndk3v1.sh

##################################################################################
1_generate_images() {
    echo " "
    echo "##################################################################################"
    echo "Step1: CREATE DATA AND FOLDERS"
    echo "##################################################################################"
    echo " "
    # clean files in pre-built sub-directories
    rm -f ${DATASET_DIR}/img_*/* ${DATASET_DIR}/seg_*/*
    # unzip the original dataset
    unzip ${WORK_DIR}/../dataset1.zip -d ${WORK_DIR}
    cd code
    # put the data into proper folders
    python prepare_data.py
    cd ..
    # clean previous directories
    rm -r ${DATASET_DIR}/annotations_* ${DATASET_DIR}/images_*
}

##################################################################################
# effective training
2_unet_train() {
    cd ${KERAS_MODEL_DIR}

    # effective training and predictions
    cd ../code
    echo " "
    echo "##################################################################################"
    echo "Step2a: TRAINING"
    echo "##################################################################################"
    echo " "
    python unet_training.py -m 1
    python unet_training.py -m 2
    python unet_training.py -m 3
    echo " "

    cd ../code
    echo "##################################################################################"
    echo "Step2b: MAKING PREDICTIONS"
    echo "##################################################################################"
    echo " "
    python unet_make_predictions.py -m 1
    python unet_make_predictions.py -m 2
    python unet_make_predictions.py -m 3
    cd ..

}

##################################################################################
# Keras to TF chkpt files
3_fcn8_Keras2TF() {
    echo " "
    echo "#######################################################################################"
    echo "Step3: KERAS to TENSORFLOW GRAPH CONVERSION"
    echo "#######################################################################################"
    echo " "
    # clean TF Check Point files
    #rm ${CHKPT_DIR}/${CNN}/*
    # from Keras to TF
    cd code
    python Keras2TF.py --model  "unet1"
    python Keras2TF.py --model  "unet2"
    python Keras2TF.py --model  "unet3"
    cd ..
}



##################################################################################
# freeze the inference graph
4a_fcn8_freeze() {
    echo " "
    echo "##############################################################################"
    echo "Step4a: FREEZE TF GRAPHS"
    echo "##############################################################################"
    echo " "
    # freeze the TF graph
    freeze_graph \
	--input_graph       ${CHKPT_DIR}/${CNN}1/${INFER_GRAPH_FILENAME} \
	--input_checkpoint  ${CHKPT_DIR}/${CNN}1/${CHKPT_FILENAME} \
	--input_binary      true \
	--output_graph      ${FREEZE_DIR}/${CNN}1/${FROZEN_GRAPH_FILENAME} \
	--output_node_names ${OUTPUT_NODE}

    # freeze the TF graph
    freeze_graph \
	--input_graph       ${CHKPT_DIR}/${CNN}2/${INFER_GRAPH_FILENAME} \
	--input_checkpoint  ${CHKPT_DIR}/${CNN}2/${CHKPT_FILENAME} \
	--input_binary      true \
	--output_graph      ${FREEZE_DIR}/${CNN}2/${FROZEN_GRAPH_FILENAME} \
	--output_node_names "conv2d_23/Relu"

    # freeze the TF graph
    freeze_graph \
	--input_graph       ${CHKPT_DIR}/${CNN}3/${INFER_GRAPH_FILENAME} \
	--input_checkpoint  ${CHKPT_DIR}/${CNN}3/${CHKPT_FILENAME} \
	--input_binary      true \
	--output_graph      ${FREEZE_DIR}/${CNN}3/${FROZEN_GRAPH_FILENAME} \
	--output_node_names ${OUTPUT_NODE}


    echo " "
    echo "##############################################################################"
    echo "Step4a: INSPECT FROZEN GRAPH"
    echo "##############################################################################"
    echo " "
    vai_q_tensorflow inspect --input_frozen_graph ${FREEZE_DIR}/${CNN}1/${FROZEN_GRAPH_FILENAME}
    vai_q_tensorflow inspect --input_frozen_graph ${FREEZE_DIR}/${CNN}2/${FROZEN_GRAPH_FILENAME}
    vai_q_tensorflow inspect --input_frozen_graph ${FREEZE_DIR}/${CNN}3/${FROZEN_GRAPH_FILENAME}
}


##################################################################################
# evaluate the original graph
4b_eval_graph() {
    echo " "
    echo "##############################################################################"
    echo "Step4b: EVALUATING THE ORIGINAL GRAPH"
    echo "##############################################################################"
    echo " "
    cd code
    python eval_graph.py \
	   --graph=../${FREEZE_DIR}/${CNN}1/${FROZEN_GRAPH_FILENAME} \
	   --input_node=${INPUT_NODE} \
	   --output_node=${OUTPUT_NODE} \
	   --gpu=0

    python eval_graph.py \
	   --graph=../${FREEZE_DIR}/${CNN}2/${FROZEN_GRAPH_FILENAME} \
	   --input_node=${INPUT_NODE} \
	   --output_node="conv2d_23/Relu" \
	   --gpu=0

    python eval_graph.py \
	   --graph=../${FREEZE_DIR}/${CNN}3/${FROZEN_GRAPH_FILENAME} \
	   --input_node=${INPUT_NODE} \
	   --output_node=${OUTPUT_NODE} \
	   --gpu=0
    cd ..
}


##################################################################################
5a_unet_quantize() {
    echo " "
    echo "##########################################################################"
    echo "Step5a: QUANTIZATION"
    echo "##########################################################################"
    echo " "
    ## use PATCHed decent_q for deconv2d_transpose layer
    #source ~/scripts/activate_py36_dnndk20191009.sh
    # log the decent_q version being used
    echo " "
    decent_q --version
    echo " "
    #quantize
    cd code
    vai_q_tensorflow quantize \
	 --input_frozen_graph  ../${FREEZE_DIR}/${CNN}1/${FROZEN_GRAPH_FILENAME} \
	 --input_nodes         ${INPUT_NODE} \
	 --input_shapes        ?,224,224,3 \
	 --output_nodes        ${OUTPUT_NODE} \
	 --output_dir          ../${QUANT_DIR}/${CNN}1/ \
	 --method              1 \
	 --input_fn            graph_input_fn.calib_input \
	 --calib_iter          10 \
	 --gpu 0

   vai_q_tensorflow quantize \
	 --input_frozen_graph  ../${FREEZE_DIR}/${CNN}2/${FROZEN_GRAPH_FILENAME} \
	 --input_nodes         ${INPUT_NODE} \
	 --input_shapes        ?,224,224,3 \
	 --output_nodes        "conv2d_23/Relu" \
	 --output_dir          ../${QUANT_DIR}/${CNN}2/ \
	 --method              1 \
	 --input_fn            graph_input_fn.calib_input \
	 --calib_iter          10 \
	 --gpu 0

   vai_q_tensorflow quantize \
	 --input_frozen_graph  ../${FREEZE_DIR}/${CNN}1/${FROZEN_GRAPH_FILENAME} \
	 --input_nodes         ${INPUT_NODE} \
	 --input_shapes        ?,224,224,3 \
	 --output_nodes        ${OUTPUT_NODE} \
	 --output_dir          ../${QUANT_DIR}/${CNN}3/ \
	 --method              1 \
	 --input_fn            graph_input_fn.calib_input \
	 --calib_iter          10 \
	 --gpu 0

    cd ..
    ## go back to official DNNDK v3.1
    #source ${HOME}/scripts/activate_py36_dnndk3v1.sh
}

##################################################################################
# make predictions with quantized graph

5b_eval_quantized_graph() {
    echo " "
    echo "##############################################################################"
    echo "Step5b: EVALUATE QUANTIZED GRAPH"
    echo "##############################################################################"
    echo " "
    cd code
    python eval_quantized_graph.py \
	   --graph=../${QUANT_DIR}/${CNN}1/${QUANTIZED_FILENAME} \
	   --input_node=${INPUT_NODE} \
	   --output_node=${OUTPUT_NODE} \
	   --gpu=0

    python eval_quantized_graph.py \
	   --graph=../${QUANT_DIR}/${CNN}2/${QUANTIZED_FILENAME} \
	   --input_node=${INPUT_NODE} \
	   --output_node="conv2d_23/Relu" \
	   --gpu=0

    python eval_quantized_graph.py \
	   --graph=../${QUANT_DIR}/${CNN}3/${QUANTIZED_FILENAME} \
	   --input_node=${INPUT_NODE} \
	   --output_node=${OUTPUT_NODE} \
	   --gpu=0

    cd ..
}



##################################################################################
# Compile ELF file
6_compile() {
  echo " "
  echo "##########################################################################"
  echo "COMPILE FCN8 ELF FILE WITH DNNDK"
  echo "##########################################################################"
  echo " "
  # log the decent_q version being used
  echo " "
  ${DNNC} --version
  echo " "
  ${DNNC} \
      --parser     tensorflow \
      --frozen_pb  ${QUANT_DIR}/${CNN}1/deploy_model.pb \
      --dcf        ${DCF_FILE} \
      --cpu_arch   arm64 \
      --output_dir ${COMPILE_DIR}/${CNN}1 \
      --save_kernel \
      --mode normal \
      --net_name   ${CNN}1

  ${DNNC} \
      --parser     tensorflow \
      --frozen_pb  ${QUANT_DIR}/${CNN}2/deploy_model.pb \
      --dcf        ${DCF_FILE} \
      --cpu_arch   arm64 \
      --output_dir ${COMPILE_DIR}/${CNN}2 \
      --save_kernel \
      --mode normal \
      --net_name   ${CNN}2


  ${DNNC} \
      --parser     tensorflow \
      --frozen_pb  ${QUANT_DIR}/${CNN}3/deploy_model.pb \
      --dcf        ${DCF_FILE} \
      --cpu_arch   arm64 \
      --output_dir ${COMPILE_DIR}/${CNN}3 \
      --save_kernel \
      --mode normal \
      --net_name   ${CNN}3


}


##################################################################################
# Compile ELF file with VAI
6_compile_vai() {
  echo " "
  echo "##########################################################################"
  echo "COMPILE UNET ELF FILE WITH Vitis AI"
  echo "##########################################################################"
  echo " "
  python /opt/vitis_ai/compiler/vai_c_tensorflow \
	 --frozen_pb ${QUANT_DIR}/${CNN}1/deploy_model.pb \
	 --arch /opt/vitis_ai/compiler/arch/dpuv2/ZCU102/ZCU102.json \
	 --output_dir ${COMPILE_DIR}/${CNN}1 \
	 --options    "{'mode':'normal'}" \
	 --net_name ${CNN}1

  python /opt/vitis_ai/compiler/vai_c_tensorflow \
	 --frozen_pb ${QUANT_DIR}/${CNN}2/deploy_model.pb \
	 --arch /opt/vitis_ai/compiler/arch/dpuv2/ZCU102/ZCU102.json \
	 --output_dir ${COMPILE_DIR}/${CNN}2 \
	 --options    "{'mode':'normal'}" \
	 --net_name ${CNN}2

  python /opt/vitis_ai/compiler/vai_c_tensorflow \
	 --frozen_pb ${QUANT_DIR}/${CNN}3/deploy_model.pb \
	 --arch /opt/vitis_ai/compiler/arch/dpuv2/ZCU102/ZCU102.json \
	 --output_dir ${COMPILE_DIR}/${CNN}3 \
	 --options    "{'mode':'normal'}" \
	 --net_name ${CNN}3
 }


##################################################################################
##################################################################################

main() {

    # clean up previous results
    rm -rf ${WORK_DIR}; mkdir ${WORK_DIR}
    rm -rf ${LOG_DIR}; mkdir ${LOG_DIR}
    rm -rf ${RPT_DIR}; mkdir ${RPT_DIR}
    rm -rf ${CHKPT_DIR}; mkdir ${CHKPT_DIR}
    rm -rf ${DATASET_DIR}; mkdir ${DATASET_DIR}
    mkdir ${DATASET_DIR}/img_calib ${DATASET_DIR}/img_test ${DATASET_DIR}/img_train ${DATASET_DIR}/img_valid
    mkdir ${DATASET_DIR}/seg_calib ${DATASET_DIR}/seg_test ${DATASET_DIR}/seg_train ${DATASET_DIR}/seg_valid
    rm -rf ${FREEZE_DIR}; mkdir ${FREEZE_DIR}
    rm -rf ${QUANT_DIR}; mkdir ${QUANT_DIR}
    rm -rf ${COMPILE_DIR}; mkdir ${COMPILE_DIR}

    mkdir ${LOG_DIR}/${CNN}
    mkdir ${CHKPT_DIR}/${CNN}1   ${CHKPT_DIR}/${CNN}2   ${CHKPT_DIR}/${CNN}3
    mkdir ${FREEZE_DIR}/${CNN}1  ${FREEZE_DIR}/${CNN}2  ${FREEZE_DIR}/${CNN}3
    mkdir ${QUANT_DIR}/${CNN}1   ${QUANT_DIR}/${CNN}2   ${QUANT_DIR}/${CNN}3
    mkdir ${COMPILE_DIR}/${CNN}1 ${COMPILE_DIR}/${CNN}2 ${COMPILE_DIR}/${CNN}3


    ## create the proper folders and images from the original dataset
    #1_generate_images 2>&1 | tee ${LOG_DIR}/${CNN}/${PREPARE_DATA_LOG}

    # do the training and make predictions
    2_unet_train     2>&1 | tee ${LOG_DIR}/${CNN}/${TRAIN_LOG}

    # from Keras to TF
    3_fcn8_Keras2TF  2>&1 | tee ${LOG_DIR}/${CNN}/unet_keras2tf.log

    # freeze the graph and inspect it
    4a_fcn8_freeze   2>&1 | tee ${LOG_DIR}/${CNN}/${FREEZE_LOG}

    # evaluate the frozen graph performance
    4b_eval_graph 2>&1 | tee ${LOG_DIR}/${CNN}/${EVAL_FR_LOG}

    # quantize with decent_q
    5a_unet_quantize 2>&1 | tee ${LOG_DIR}/${CNN}/${QUANT_LOG}

    # evaluate post-quantization model
    5b_eval_quantized_graph 2>&1 | tee ${LOG_DIR}/${CNN}/${EVAL_Q_LOG}

    # compile with dnnc to generate elf file
    ##6_compile 2>&1 | tee ${LOG_DIR}/${CNN}/${COMP_LOG}
    6_compile_vai 2>&1 | tee ${LOG_DIR}/${CNN}/${COMP_LOG}

    # move elf and so files to target board directory
    mv  ${COMPILE_DIR}/${CNN}1/dpu*.elf    ${TARGET_DIR}/v1/model/
    mv  ${COMPILE_DIR}/${CNN}2/dpu*.elf    ${TARGET_DIR}/v2/model/
    mv  ${COMPILE_DIR}/${CNN}3/dpu*.elf    ${TARGET_DIR}/v3/model/

: '
    # copy test images into target board
    tar -cvf "test.tar" ${DATASET_DIR}/img_test ${DATASET_DIR}/seg_test
    gzip test.tar
    mv test.tar.gz ${TARGET_DIR}/../
'

    echo "#####################################"
    echo "MAIN FLOW COMPLETED"
    echo "#####################################"


}

main
