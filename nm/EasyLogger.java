package nm;

import java.text.SimpleDateFormat;
import java.util.Calendar;

public class EasyLogger {
	public static void log(String message){
		System.out.println(message + ": " + now());
	}
	
	public static String now() {
	    Calendar cal = Calendar.getInstance();
	    SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
	    return sdf.format(cal.getTime());
	}
}
