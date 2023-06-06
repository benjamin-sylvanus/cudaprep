# all: test
# nvcc  -arch=sm_75 -ccbin g++ -dc -m64 -o
# nvcc -arch=sm_75 -ccbin g++ -dc -L /usr/local/cuda/lib64 main.cu -lcublas -lcurand
# test:

srcdir = src
bindir = bin
staticLibs = -lcublas -lcurand -lcusparse
cudaLib = /usr/local/cuda/lib64
objects =  ./bin/datatemplate.o ./bin/simulation.o ./bin/simreader.o ./bin/controller.o ./bin/viewer.o ./bin/funcs.o ./bin/overloads.o
all: $(objects)
		nvcc -arch=sm_75 -ccbin g++ -dc -L$(cudaLib) main.cu $(staticLibs) -o ./bin/main.o 
		nvcc  -arch=sm_75 -dlink $(objects)  ./bin/main.o -o ./bin/gpuCode.o
		nvcc  -arch=sm_75 ./bin/main.o ./bin/gpuCode.o $(objects) -o app

$(bindir)/%.o: $(srcdir)/%.cpp
		nvcc -x cu -arch=sm_75 -I. -dc $< -o $@

some:
		nvcc -arch=sm_75 -ccbin g++ -dc -m64 -o ./bin/main.o -c main.cu
		nvcc -arch=sm_75 ./bin/main.o  $(objects) -o app

clean:
		rm -f ./bin/*.o
		rm -f *.o app

cleanapp:
		rm -f app
