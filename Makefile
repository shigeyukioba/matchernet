DIR = ./bundlenet004
DIR5 = ./bundlenet005
DIR6 = ./bundlenet006
DIRJJ = ./bundlenet_jae
BDIR = ../../BriCA1/python/brica1
BOBJS = $(BDIR)/component.py $(BDIR)/unit.py $(BDIR)/scheduler.py $(BDIR)/utils.py
#OBJS = $(DIR)/bundlenet.py $(DIR)/state.py $(DIR)/provider.py $(DIR)/EKF.py $(DIR)/recorder.py $(DIR)/physics.py $(DIR)/app/writing.py
OBJS = $(DIR)/bundlenet.py $(DIR)/state.py $(DIR)/provider.py $(DIR)/EKF.py
OBJS5 = $(DIR5)/bundlenet.py $(DIR5)/state.py $(DIR5)/brica1/scheduler.py $(DIR5)/brica1/component.py $(DIR5)/brica1/connection.py
OBJSJJ = $(DIRJJ)/EKF.py $(DIRJJ)/bundlenet.py $(DIRJJ)/state.py $(DIRJJ)/matcher.py $(DIRJJ)/fn.py
OBJS6 = $(DIR6)/bundlenet.py $(DIR6)/provider.py $(DIRJJ)/fn.py $(DIR6)/state.py $(DIR6)/utils.py $(DIR6)/ekf.py $(DIR6)/ekf_test.py $(DIR6)/ekf_test_multiple_provider.py
ps004:
	enscript --highlight=python --color --landscape --columns=2 --tabsize=2 --fancy-header --borders -o source.ps $(OBJS);

ps006:
	enscript --highlight=python --color --landscape --columns=2 --tabsize=2 --fancy-header --borders -o source_006.ps $(OBJS6);

psbrica1:
	enscript --highlight=python --color --landscape --columns=2 --tabsize=2 --fancy-header --borders -o source_brica1.ps $(BOBJS);

psbrica5:
	enscript --highlight=python --color --landscape --columns=2 --tabsize=2 --fancy-header --borders -o source_brica5.ps $(OBJS5);

psjj:
	enscript --highlight=python --color --landscape --columns=2 --tabsize=2 --fancy-header --borders -o source_jj.ps $(OBJSJJ);
