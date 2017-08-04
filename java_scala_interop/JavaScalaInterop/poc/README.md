To run a maven Scala project:

Approach 1: using maven

mvn package exec:java -Dexec.mainClass=org.vario.demo.App

If you don't want to pass mainClass manually, you can do this in plugin configuration:

<plugins>
  <plugin>
    <groupId>org.codehaus.mojo</groupId>
    <artifactId>exec-maven-plugin</artifactId>
    <version>1.1</version>
    <configuration>
      <mainClass>com.example.Main</mainClass>
    </configuration>
  </plugin>
</plugins>


Approach 2: using the java command  

maven-jar-plugin

If you want to ship the application, use maven-jar-plugin to add Main-Class and Class-Path entries to the manifest:

Main-Class: com.example.Main
Class-Path: lib/scala-library-2.9.0-1.jar lib/slf4j-api-1.6.1.jar ...
The following configuration does that and also copies all the dependencies (including Scala runtime library) to target/lib.

<plugin>
    <artifactId>maven-jar-plugin</artifactId>
    <version>2.3.1</version>
    <configuration>
        <archive>
            <manifest>
                <mainClass>com.example.Main</mainClass>
                <addClasspath>true</addClasspath>
                <classpathLayoutType>custom</classpathLayoutType>
                <customClasspathLayout>lib/$${artifact.artifactId}-$${artifact.version}$${dashClassifier?}.$${artifact.extension}
                </customClasspathLayout>
            </manifest>
        </archive>
    </configuration>
</plugin>
<plugin>
    <artifactId>maven-dependency-plugin</artifactId>
    <version>2.3</version>
    <configuration>
        <outputDirectory>${project.build.directory}/lib</outputDirectory>
    </configuration>
    <executions>
        <execution>
            <phase>package</phase>
            <goals>
                <goal>copy-dependencies</goal>
            </goals>
        </execution>
    </executions>
</plugin>
Now you can simply run your application by (note the target/lib directory is required):

$ java -jar target/your_app-VERSION.jar
You can ship your application simply by copying your JAR file along with /lib subdirectory.

Also see Exec Maven Plugin and Playing with Scala and Maven.

Reference: https://stackoverflow.com/questions/6758258/running-a-maven-scala-project
