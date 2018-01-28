CC = mpicc
CXX = mpicxx
CFLAGS = -O3 -std=gnu99
CXXFLAGS = -O3 -std=gnu++11

STUDENTID = $(USER:p%=%)
TARGETS = HW1_$(STUDENTID)_basic HW1_$(STUDENTID)_advanced

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean: 
	rm -f $(TARGETS)
