field_eval :
	gcc -O3 -fPIC -shared field_eval.c arc_field_eval.c gradient_eval.c arc_gradient_eval.c -lm -lgsl -lgslcblas -o field_eval.so
