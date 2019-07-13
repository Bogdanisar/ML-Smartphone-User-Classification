.SUFFIXES:
.PRECIOUS: %.exe %.cpp %.in %.out


%: %.in %.out %.exe
	# ./$*.exe < $*.in > $*.out 2>&1
	./$*.exe < $*.in > $*.out


%.cpp: 
	cp default_main.cpp $@

%.in: 
	touch $@

%.out: 
	touch $@

%.exe: %.cpp
	g++ -g -std=c++17 -Wall -Wextra -pedantic $*.cpp -o $*.exe

