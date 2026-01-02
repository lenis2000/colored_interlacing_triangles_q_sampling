CXX = g++
CXXFLAGS = -O3 -std=c++17 -Wall -march=native

all: sampler sampler_collect

sampler: sampler.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

sampler_collect: sampler_collect.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f sampler sampler_collect

test: sampler
	./sampler 3 0.5 100000
	./sampler 5 1.0 100000

.PHONY: all clean test
