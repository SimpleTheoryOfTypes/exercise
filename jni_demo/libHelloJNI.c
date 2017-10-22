#include <jni.h>
#include <stdio.h>
#include "HelloJNI.h"

JNIEXPORT void JNICALL Java_HelloJNI_sayHello(JNIEnv *env, jobject thisObj) {
  printf("[Jiading Gai] Do something useful with Java Native Interface (JNI)!\n");
  return;
}
