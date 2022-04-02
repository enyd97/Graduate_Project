gemv :gemv_host.c
	gcc --std=c99 -o mm mm.c `dpu-pkg-config --cflags --libs dpu`

clean :
	rm mm