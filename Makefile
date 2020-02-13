JL          = ~/.julia/julia
BASE        = functions.jl setup.jl
CORE        = ml_core.jl
OBJS        = main.jl
INITS       = initialize.jl

main: $(BASE) $(CORE) $(OBJS)
	$(JL) $(OBJS)

init: $(BASE) $(INITS)
	$(JL) $(INITS)

clean:
	-rm -f *.txt *.png *.dat
	-rm -rf data
