all:HelloJNI.class HelloJNI.h libHelloJNI.so
JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/
HelloJNI.class:HelloJNI.java
	javac $^
HelloJNI.h:HelloJNI.java
	javah -cp . HelloJNI
libHelloJNI.so:libHelloJNI.c HelloJNI.h
	gcc -shared -fPIC -I ${JAVA_HOME}/include -I ${JAVA_HOME}/include/linux $^ -o $@
run:HelloJNI.class libHelloJNI.so
	java -cp . -Djava.library.path=./ HelloJNI
clean:
	rm -f *.class *.h *.so
