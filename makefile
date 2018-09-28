field_eval :
	gcc -O3 -fPIC -shared field_eval.c arc_field_eval.c pgradient_eval.c -lm -lgsl -lgslcblas -o field_eval.so
