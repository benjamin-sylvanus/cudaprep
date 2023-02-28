# all: test 

# test:
-include objects.mk

objects = ./src/particle.o ./src/datatemplate.o ./src/simulation.o ./src/simreader.o
all: $(objects)
		nvcc -ccbin g++ -dc -m64 -o main.o -c main.cu
		nvcc -arch=sm_75 -dlink $(objects) main.o -lcublas -lcurand -o gpuCode.o
		nvcc -arch=sm_75 gpuCode.o main.o  $(objects) -o app

		
%.o: %.cpp
		nvcc -x cu -arch=sm_75 -I. -dc $< -o $@

some: 
		nvcc -ccbin g++ -dc -m64 -o main.o -c main.cu
		nvcc -arch=sm_75 main.o  $(objects) -o app
clean:
		rm -f ./src/*.o
		rm -f *.o app
		# rm cublas-example
cleanapp:
		rm -f app

