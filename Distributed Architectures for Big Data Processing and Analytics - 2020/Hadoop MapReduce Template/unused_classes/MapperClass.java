package it.polito.bigdata.hadoop;

import java.io.IOException;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Mapper;
import java.net.URI;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
//import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs; //-> MultipleOutput Case

//InputFormat: TextInputFormat.class (LongWritable Key - Text Value),
//				KeyValueTextInputFormat.class (Text Key - Text Value),
//				...
//OutputFormat: TextOutputFormat.class (format = key\tValue\n)...
//(K-V)DataTypeClass: Text.class, IntWritable.class, LongWritable.class...
//(K-V)DataType: Text, IntWritable, LongWritable, DoubleWritable...

class MapperClass extends Mapper<..., //-> Input K-DataType
								..., //-> Input V-DataType
								..., //-> Output K-DataType
								...> //-> Output V-DataType
								{
									
	//private int min = Integer.MIN_VALUE;
	//private double max = Double.MAX_VALUE;
	//int property; //-> Property Case
	
	//private MultipleOutputs<Text, NullWritable> mos = null; //-> MultipleOutput Case
	
	// Structures:
	//private WritableClass[] ArrayName = new WritableClass[12]; //-> CustomClass array
	//private HashMap<K-Type, V-Type> hashMapName;
	//private HashSet<String> hashSetName;
	//private ArrayList<String> arrayListName;

	protected void setup(Context context) { //throws IOException, InterruptedException //-> DistributedCache Case
		//property = Integer.parseInt(context.getConfiguration().get("Property Name")); //-> Property Case
		//mos = new MultipleOutputs<key type, value type>(context); //-> MultipleOutput Case
		//context.getCounter(COUNTERS_GROUP.COUNTER1_NAME).increment(1); //-> Counter Case
		
		//Structures:
		//hashMapName = new HashMap<K-Type, V-Type>();
		//hashSetName = new HashSet<String>();
		//arrayListName = new ArrayList<String>();
		//for (int i = 0; i < ArrayName.length; i++) { //-> CustomClass array
		//ArrayName[i] = new WritableClass(); 
		//}
		
		// DistributedCache Case
		//String line;
		//URI[] urisCachedFiles = context.getCacheFiles();
		//BufferedReader file = new BufferedReader(new FileReader(new File(urisCachedFiles[0].getPath())));
		//while ((line = file.readLine()) != null) {
		//		process one line per cycle...
		//		es. arrayListName.add(nextLine);
		//}
		//file.close();
	}

	protected void map(... key, //-> Input K-DataType,
					   ... value, //-> Input V-DataType,
					   Context context) throws IOException, InterruptedException {

		//String[] words = value.toString().replaceAll("\\p{Punct}","").toLowerCase().split("\\s+");
        //for(String word : words){
			//word = word.substring(0, word.length() - 1);
			//if(word.compareTo("Ciao") == 0) ...
			//if(word.contains("sottostringa") == true)...
			//if(word.startsWith("He") == true)...
			//word = word.replace("fox", "dog"));
			//context.write(new Text(word), new IntWritable(1));
        //}
		
		//Gestione strutture
		// HASHSET
		//hashSetName.add("India");
		//hashSetName.remove("Australia");
		//if (hashSetName.contains("vishal") == true)...
		//Iterator<String> i = hashSetName.iterator(); 
        //while (i.hasNext()){
			//System.out.println(i.next()); 
		//}
		//size = hashSetName.size()
		
		// HASHMAP
		//hashMapName.put(key, value);
		//hashMapName.remove(key);
		//size = hashMapName.size()
		//if (hashMapName.containsKey("vishal") == true)...
        //Integer a = hashMapName.get(key)
            
		
		// MultipleOutput Case
		//if(condition){
		//	mos.write("prefix1", new Text(...), NullWritable.get());
		//} else {
		//	mos.write("normaltemp", new Text(...), NullWritable.get());
		//}

	}

	protected void cleanup(Context context) throws IOException, InterruptedException {		
		
		//mos.close(); //-> MultipleOutput Case
		
		//for (Entry<K-Type, V-Type> pair : hashMapName.entrySet()) {
		//	context.write(new K-DataType(pair.getKey()), new V-DataType(pair.getValue()));
		//}
	}

}
