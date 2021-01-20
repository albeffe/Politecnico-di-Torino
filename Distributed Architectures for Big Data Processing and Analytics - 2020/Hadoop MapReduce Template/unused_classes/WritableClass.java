package it.polito.bigdata.hadoop;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

//InputFormat: TextInputFormat.class (LongWritable Key - Text Value),
//				KeyValueTextInputFormat.class (Text Key - Text Value),
//				...
//OutputFormat: TextOutputFormat.class (format = key\tValue\n)...
//(K-V)DataTypeClass: Text.class, IntWritable.class, LongWritable.class...
//(K-V)DataType: Text, IntWritable, LongWritable, DoubleWritable...

public class WritableClass implements org.apache.hadoop.io.Writable {

	private double value;
	private int cnt;

	public WritableClass() {
		super();
		this.value = 0.0;
		this.cnt = 0;
	}

	public double getValue() {
		return value;
	}

	public void setValue(double value) {
		this.value = value;
	}
	
	public void incrementValue(double income) {
		double val = this.value;
		this.value = val + income;
	}

	public int getCnt() {
		return cnt;
	}

	public void setCnt(int cnt) {
		this.cnt = cnt;
	}
	
	public void incrementCnt() {
		int cn = this.cnt;
		this.cnt = cn + 1;
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		value = in.readDouble();
		cnt = in.readInt();
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeDouble(value);
		out.writeInt(cnt);
	}

	public String toString() {
		String formattedString = new String("value:" + value + " cnt:" + cnt);

		return formattedString;
	}

}
