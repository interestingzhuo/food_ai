all: undistort.so

clean:
	rm -f *.so *.o

undistort.so: undistort.c

	cc -fPIC -std=c99 -shared -o undistort.so undistort.c
