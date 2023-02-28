################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../main.cu 

O_SRCS += \
../main.o 

OBJS += \
./main.o 

CU_DEPS += \
./main.d 

CUDA_DIR = /usr/local/cuda/ 
SAMPLES_DIR = CUDA_DIR + samples/

# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -I"$(SAMPLES_DIR)7_CUDALibraries" -I"$(SAMPLES_DIR)common/inc" -I"/autofs/homes/009/bs244/cuda-workspace/hellocuda/cudaprep(copy)/cudaprep/" -G -g -O0 -gencode arch=compute_75,code=sm_75  -odir "." -M -o "$(@:%.o=%.d)" "$<"
	nvcc -I"$(SAMPLES_DIR)7_CUDALibraries" -I"$(SAMPLES_DIR)common/inc" -I"/autofs/homes/009/bs244/cuda-workspace/hellocuda/cudaprep(copy)/cudaprep/" -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_75,code=sm_75  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


