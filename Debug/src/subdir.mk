################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/datatemplate.cpp \
../src/particle.cpp \
../src/simreader.cpp \
../src/simulation.cpp 

OBJS += \
./src/datatemplate.o \
./src/particle.o \
./src/simreader.o \
./src/simulation.o 

CPP_DEPS += \
./src/datatemplate.d \
./src/particle.d \
./src/simreader.d \
./src/simulation.d 

CUDA_DIR = /usr/local/cuda/
SAMPLES_DIR = $(CUDA_DIR)samples/

# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -I"$(SAMPLES_DIR)7_CUDALibraries" -I"$(SAMPLES_DIR)common/inc" -I"/autofs/homes/009/bs244/cuda-workspace/hellocuda/cudaprep(copy)/cudaprep/" -G -g -O0 -gencode arch=compute_75,code=sm_75  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	nvcc -I"$(SAMPLES_DIR)7_CUDALibraries" -I"$(SAMPLES_DIR)common/inc" -I"/autofs/homes/009/bs244/cuda-workspace/hellocuda/cudaprep(copy)/cudaprep/" -G -g -O0 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


