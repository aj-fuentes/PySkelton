default : field_eval_static

field_eval : field_eval.c arc_field_eval.c gradient_eval.c arc_gradient_eval.c
	gcc -O3 -fPIC -shared field_eval.c arc_field_eval.c gradient_eval.c arc_gradient_eval.c -lm -lgsl -lgslcblas -o field_eval.so

field_eval_static : field_eval.c arc_field_eval.c gradient_eval.c arc_gradient_eval.c
	gcc -I ./include -O3 -fPIC -shared -o field_eval_static.so field_eval.c arc_field_eval.c gradient_eval.c arc_gradient_eval.c -lm libgsl.a
