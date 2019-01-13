DIR = ./matchernet_py_001
OBJS = $(DIR)/matchernet.py $(DIR)/observer.py $(DIR)/ekf.py $(DIR)/state_space_model_2d.py $(DIR)/ekf_test.py $(DIR)/ekf_test_multiple_observers.py $(DIR)/test_bundlenet_with_brica2.py $(DIR)/fn.py $(DIR)/utils.py $(DIR)/state.py $(DIR)/matchernet_null.py

psmn001:
	enscript --highlight=python --color --landscape --columns=2 --tabsize=2 --fancy-header --borders -o ./doc/source_mn.ps $(OBJS);
