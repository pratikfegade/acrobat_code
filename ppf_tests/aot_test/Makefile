CXX=g++
CXX_FLAGS=-w -O3 -I${TVM_HOME}/include -I${TVM_HOME}/3rdparty/dlpack/include/ -I${TVM_HOME}/3rdparty/dmlc-core/include/ -L${TVM_HOME}/build/ -Wl,-rpath=${TVM_HOME}/build/ -fopenmp
LD_FLAGS=-ltvm_runtime

%: %_src.cpp %_src.hpp %_main.cpp
	@${CXX} -o $@ ${CXX_FLAGS} $@_main.cpp $< ${LD_FLAGS}

clean:
	@rm -f aot_test treelstm *.o
