package it.polito.bigdata.hadoop;

import java.io.IOException;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Reducer;
//import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs; //-> MultipleOutput Case

//InputFormat: TextInputFormat.class (LongWritable Key - Text Value),
//				KeyValueTextInputFormat.class (Text Key - Text Value),
//				...
//OutputFormat: TextOutputFormat.class (format = key\tValue\n)...
//(K-V)DataTypeClass: Text.class, IntWritable.class, LongWritable.class...
//(K-V)DataType: Text, IntWritable, LongWritable, DoubleWritable...

class CombinerClass extends Reducer<..., //-> Input K-DataType
									..., //-> Input V-DataType
									..., //-> Output K-DataType
									...> //-> Output V-DataType
									{
										
	//private int min = Integer.MIN_VALUE;
	//private double max = Double.MAX_VALUE;
	//private WritableClass[] months = new WritableClass[12];
	//for (int i = 0; i < months.length; i++) {
		//months[i] = new WritableClass(); 
	//}

	protected void setup(Context context) {
		//int variabile1 = Integer.parseInt(context.getConfiguration().get("Property Name")); //-> Property Case
		//mos = new MultipleOutputs<key type, value type>(context); //-> MultipleOutput Case
		//URI[] urisCachedFiles = context.getCacheFiles(); //-> DistributedCache Case
		//context.getCounter(COUNTERS_GROUP.COUNTER1_NAME).increment(1); //-> Counter Case		
	}

	@Override
	protected void reduce(... key, //-> Input K-DataType,
						  Iterable<...> values, //-> Input V-DataType
						  Context context) throws IOException, InterruptedException {

		// Example
		//String[] parts = value.toString().split("\\s+");
        //for(String part : parts){
        //    String cleanedPart = part.toLowerCase();
        //    context.write(new Text(cleanedPart), new IntWritable(1));
        //}
		
		//Gestione strutture
		// HASHSET
		//hashSetName.add("India");
		//hashSetName.remove("Australia");
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
		
		// DistributedCache Case
		//BufferedReaderfile = new BufferedReader(new FileReader(new File(urisCachedFiles [0]. getPath())));
		//while ((line = file.readLine()) != null) {
		//	// process one line per cycle
		//	...
		//}
		//file.close();
	}
	
	protected void cleanup(Context context) throws IOException, InterruptedException {
		
		//mos.close(); //-> MultipleOutput Case
		
		//for (Entry<K-Type, V-Type> pair : hashMapName.entrySet()) {
		//	context.write(new K-DataType(pair.getKey()), new V-DataType(pair.getValue()));
		//}
	}
}