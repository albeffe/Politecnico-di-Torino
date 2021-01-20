package it.polito.bigdata.hadoop;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.*;
import org.apache.hadoop.mapreduce.lib.output.*;
//import org.apache.hadoop.mapreduce.Counter; //-> Counter Case
//import org.apache.hadoop.util.Time; //-> Counter Case
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

//InputFormat: TextInputFormat.class (LongWritable Key - Text Value),
//				KeyValueTextInputFormat.class (Text Key - Text Value),
//				...
//OutputFormat: TextOutputFormat.class (format = key\tValue\n)...
//(K-V)DataTypeClass: Text.class, IntWritable.class, LongWritable.class...
//(K-V)DataType: Text, IntWritable, LongWritable, DoubleWritable...

public class DriverClass extends Configured implements Tool {
	
	//public static enum COUNTERS_GROUP {
	//	COUNTER1_NAME,
	//	COUNTER2_NAME
	//} //-> Counter case

	@Override
	public int run(String[] args) throws Exception {

		int exitCode;
		int numberOfReducersJob_1 = Integer.parseInt(args[0]);
		Path inputDir1 = new Path(args[1]);
		// Path inputPath2 = new Path(args[2]); //-> MultipleInputs case
		Path outputDir = new Path(args[2]);

		Configuration conf = this.getConf();
		//conf.set("Property Name", "Property Value"); //-> String Name, String Value
		Job job_1 = Job.getInstance(conf);
		job_1.setJobName("Job_1 Name"); //->Insert: Job_1 Name
		//job_1.addCacheFile(new Path(args[4]).toUri()); //-> File path or Name if File in project folder
		FileInputFormat.addInputPath(job_1, inputDir1); //-> To be commented if MultipleInputs
		FileOutputFormat.setOutputPath(job_1, outputDir);
		
		job_1.setJarByClass(DriverClass.class);
		job_1.setInputFormatClass(...); //->Insert: InputFormat, To be commented if MultipleInputs
		//MultipleInputs.addInputPath(job_1, inputDir1, ..., ...); //-> InputFormat, MapperClassName.class
		//MultipleInputs.addInputPath(job_1, inputDir2, ..., ...); //-> InputFormat, MapperClassName.class
		job_1.setOutputFormatClass(...); //-> OutputFormat
		//MultipleOutputs.addNamedOutput(job_1, "no_underscores_here", ..., ..., ...); //-> Prefix, OutputFormat, K-DataTypeClass, V-DataTypeClass
		//MultipleOutputs.addNamedOutput(job_1, "no_underscores_here", ..., ..., ...); //-> Prefix, OutputFormat, K-DataTypeClass, V-DataTypeClass

		job_1.setMapperClass(...); //-> MapperClassName.class, To be commented if MultipleInputs
		job_1.setMapOutputKeyClass(...); //-> K-DataTypeClass, To be commented if MultipleOutputs if it's a map only job
		job_1.setMapOutputValueClass(...); //-> V-DataTypeClass, To be commented if MultipleOutputs if it's a map only job
		
		//job_1.setCombinerClass(...); //-> CombinerClassName.class

		job_1.setReducerClass(...); //-> ReducerClassName.class
		job_1.setOutputKeyClass(...); //-> K-DataTypeClass, To be commented if MultipleOutputs
		job_1.setOutputValueClass(...); //-> V-DataTypeClass, To be commented if MultipleOutputs

		job_1.setNumReduceTasks(numberOfReducersJob_1);

		if (job_1.waitForCompletion(true) == true) {
			
			exitCode = 0;
			//Counter myCounter = job_1.getCounters().findCounter(COUNTERS_GROUP.COUNTER1_NAME); //-> Counter case
			//System.out.println("Counter value = " + myCounter.getValue()); //-> Counter case
			
			//Job job_2 = Job.getInstance(conf);
			//job_2.setJobName("Job_2 Name"); //-> Job_2 Name
			//int numberOfReducersJob_2 = Integer.parseInt(args[5]);
			//...

			//if (job_2.waitForCompletion(true) == true)
				//exitCode = 0;
			//else
				//exitCode = 1;
		} else
			exitCode = 1;

		return exitCode;
	}

	public static void main(String args[]) throws Exception {
		int res = ToolRunner.run(new Configuration(), new DriverClass(), args);
		System.exit(res);
	}
}
