package org.vario.demo

/**
 * @author ${user.name}
 */
object App {
  
  def foo(x : Array[String]) = x.foldLeft("")((a,b) => a + b)
  
  def main(args : Array[String]) {
    println( "[JD] Hello World!" )
    println("concat arguments = " + foo(args))

    val Obj = new JTraitImpl("JD's Trait!")
    println("[JD] " + Obj.upperTraitName)
  }

}
