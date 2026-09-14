import com.sun.tools.attach.VirtualMachine;
import java.io.File;

public class AttachSingle {
    public static void main(String[] args) throws Exception {
        String pid = args[0];
        String agentJar = new File(args[1]).getAbsolutePath();
        String classFile = new File(args[2]).getAbsolutePath();
        System.out.println("Attaching to PID " + pid + "...");
        VirtualMachine vm = VirtualMachine.attach(pid);
        System.out.println("Loading agent: " + agentJar + " with args: " + classFile);
        vm.loadAgent(agentJar, classFile);
        vm.detach();
        System.out.println("Done!");
    }
}
