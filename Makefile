JL          = ~/.julia/julia
BASE        = functions.jl setup.jl
CORE        = ml_core.jl
OBJS        = main.jl
CALC        = calculation.jl
MAIL        = ./mail.sh

main: $(BASE) $(CORE) $(OBJS) $(CALC) $(MAIL)
	$(JL) $(OBJS)
	$(JL) $(CALC)
	$(MAIL)

init: $(BASE) $(INITS)
	$(JL) $(INITS)

clean:
	-rm -f *.txt *.png *.dat
	-rm -rf data error
