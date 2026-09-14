import com.sun.tools.attach.VirtualMachine;
import java.io.File;

public class Attach {
    public static void main(String[] args) throws Exception {
        String pid = args[0];
        String agentJar = new File(args[1]).getAbsolutePath();
        String className = args[2];
        String classFile = new File(args[3]).getAbsolutePath();
        String agentArgs = className + "|" + classFile;
        System.out.println("Attaching to PID " + pid + "...");
        VirtualMachine vm = VirtualMachine.attach(pid);
        System.out.println("Loading agent: " + agentJar + " with args: " + agentArgs);
        vm.loadAgent(agentJar, agentArgs);
        vm.detach();
        System.out.println("Done!");
    }
}
