gemv :gemv_host.c
	gcc --std=c99 -o gemv gemv_host.c `dpu-pkg-config --cflags --libs dpu`

clean :
	rm vector_addition