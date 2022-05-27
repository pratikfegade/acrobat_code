CXX=g++
CXX_FLAGS=-w -O3 -I${TVM_HOME}/include -I${TVM_HOME}/3rdparty/dlpack/include/ -I${TVM_HOME}/3rdparty/dmlc-core/include/ -L${TVM_HOME}/build/ -Wl,-rpath=${TVM_HOME}/build/ -fopenmp
LD_FLAGS=-ltvm_runtime -lboost_fiber -lboost_context

%_small: %_small_src.cpp %_small_src.hpp %_main.cpp array_utils.hpp
	@${CXX} -o $@ ${CXX_FLAGS} ${MODEL_SIZE_VARS} ${EXECUTOR_VARS} $*_main.cpp $< ${LD_FLAGS}

%_large: %_large_src.cpp %_large_src.hpp %_main.cpp array_utils.hpp
	@${CXX} -o $@ ${CXX_FLAGS} ${MODEL_SIZE_VARS} ${EXECUTOR_VARS} $*_main.cpp $< ${LD_FLAGS}

clean:
	@rm -f *.o treelstm_small mvrnn_small birnn_small bertxit_small graph_gen_small drnn_small lstm_cell_small
	@rm -f *.o treelstm_large mvrnn_large birnn_large bertxit_large graph_gen_large drnn_large lstm_cell_large
	@rm -f *.o treelstm mvrnn birnn bertxit graph_gen drnn lstm_cell
