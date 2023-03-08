all: Sor.so
# -g -fsanitize=address -fno-omit-frame-pointer 

# ---- Link ---------------------------
Sor.so:  solver_py.o image.o solver.o
	gcc -shared solver_py.o image.o solver.o -o Sor.so

# ---- gcc C compile ------------------
solver_py.o: solver_py.c
	gcc -fPIC -c solver_py.c -I/usr/include/python3.6m -I/usr/lib64/python3.6/site-packages/numpy/core/include/
	
image.o: image.c
	gcc -fPIC -c image.c

solver.o: solver.c
	gcc -fPIC -c solver.c
