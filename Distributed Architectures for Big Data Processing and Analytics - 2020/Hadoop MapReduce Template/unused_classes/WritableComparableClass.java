// BooleanWritable, BytesWritable, ByteWritable, DoubleWritable,
// FloatWritable, ID, ID, IntWritable, JobID, JobID, LongWritable,
// MD5Hash, NullWritable, Record, ShortWritable, TaskAttemptID,
// TaskAttemptID, TaskID, TaskID, Text, VIntWritable, VLongWritable

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

public class WritableComparableClass implements org.apache.hadoop.io.WritableComparable<WritableComparableClass> {
    
    private int counter;
    private long timestamp;
	
	public WritableComparableClass(){
		this.counter = 0;
		this.timestamp = 0;
	}

	public int getDate() {
		return counter;
	}

	public void setDate(int counter) {
		counter = counter;
	}

	public long getIncome() {
		return timestamp;
	}

	public void setIncome(long timestamp) {
		timestamp = timestamp;
	}
       
    public void readFields(DataInput in) throws IOException {
        counter = in.readInt();
		timestamp = in.readLong();
    }
	
	public void write(DataOutput out) throws IOException {
		out.writeInt(counter);
        out.writeLong(timestamp);
    }
       
    public int compareTo(WritableComparableClass o) {
        int thisValue = this.counter;
        int thatValue = o.counter;
        return (thisValue < thatValue ? -1 : (thisValue==thatValue ? 0 : 1));
    }

    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + counter;
        result = prime * result + (int) (timestamp ^ (timestamp >>> 32));
        return result
    }
}