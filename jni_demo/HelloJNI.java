public class HelloJNI {

  static { System.loadLibrary("HelloJNI"); }

  private native void sayHello();

  public static void main(String[] args) {
    new HelloJNI().sayHello();
  }
}
