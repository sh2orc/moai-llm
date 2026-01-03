use anyhow::{Context, Result};
use arrow::array::{ArrayRef, Int32Array, ListArray, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use rayon::prelude::*;
use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(name = "moai-tokenizer")]
#[command(about = "Ultra-fast Rust tokenizer for MOAI-LLM (40,000+ samples/sec)", long_about = None)]
struct Args {
    /// Input Parquet file
    #[arg(short, long)]
    input: PathBuf,

    /// Output Parquet file
    #[arg(short, long)]
    output: PathBuf,

    /// Tokenizer JSON file
    #[arg(short, long)]
    tokenizer: PathBuf,

    /// Text column name
    #[arg(long, default_value = "text")]
    text_column: String,

    /// Max sequence length (0 = no truncation, for packing mode)
    #[arg(long, default_value = "0")]
    max_length: usize,

    /// Batch size
    #[arg(short, long, default_value = "10000")]
    batch_size: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("================================================================================");
    println!("🚀 MOAI Ultra-Fast Rust Tokenizer");
    println!("================================================================================");
    println!("Input:        {}", args.input.display());
    println!("Output:       {}", args.output.display());
    println!("Tokenizer:    {}", args.tokenizer.display());
    println!("Text column:  {}", args.text_column);
    println!("Max length:   {}", if args.max_length == 0 { "No truncation (packing mode)".to_string() } else { args.max_length.to_string() });
    println!("Batch size:   {}", args.batch_size);
    println!("Threads:      {}", rayon::current_num_threads());
    println!("================================================================================");

    // Load tokenizer
    println!("📚 Loading tokenizer...");
    let tokenizer = Arc::new(
        Tokenizer::from_file(&args.tokenizer)
            .context("Failed to load tokenizer")?
    );
    println!("✅ Tokenizer loaded: vocab_size={}", tokenizer.get_vocab_size(true));

    // Open input Parquet
    println!("📂 Loading input dataset...");
    let file = File::open(&args.input)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let total_rows = builder.metadata().file_metadata().num_rows() as usize;
    println!("✅ Dataset loaded: {} samples", total_rows);

    let reader = builder.build()?;

    // Output schema
    let output_schema = Arc::new(Schema::new(vec![
        Field::new(
            "input_ids",
            DataType::List(Arc::new(Field::new("item", DataType::Int32, false))),
            false
        ),
    ]));

    // Create output writer
    let output_file = File::create(&args.output)?;
    let props = WriterProperties::builder()
        .set_compression(parquet::basic::Compression::SNAPPY)
        .build();
    let mut writer = ArrowWriter::try_new(output_file, output_schema.clone(), Some(props))?;

    // Progress bar
    let pb = ProgressBar::new(total_rows as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({per_sec}) {msg}")
            .unwrap()
            .progress_chars("=>-"),
    );

    println!("🔤 Tokenizing with Rayon parallel processing...");
    let start_time = std::time::Instant::now();
    let mut total_processed = 0;

    // Process batches
    for batch_result in reader {
        let batch = batch_result?;

        // Extract text column
        let text_array = batch
            .column_by_name(&args.text_column)
            .context(format!("Column '{}' not found", args.text_column))?
            .as_any()
            .downcast_ref::<StringArray>()
            .context("Text column must be StringArray")?;

        // Convert to Vec<String>
        let texts: Vec<String> = (0..text_array.len())
            .map(|i| text_array.value(i).to_string())
            .collect();

        // Parallel tokenization with Rayon (key performance boost!)
        let tokenized: Vec<Vec<i32>> = texts
            .par_chunks(args.batch_size)
            .flat_map(|chunk| {
                let tokenizer = Arc::clone(&tokenizer);
                chunk
                    .iter()
                    .map(|text| {
                        let encoding = tokenizer
                            .encode(text.clone(), true)
                            .expect("Tokenization failed");

                        let ids = encoding.get_ids();

                        // Apply truncation if max_length > 0
                        if args.max_length > 0 && ids.len() > args.max_length {
                            ids[..args.max_length].iter().map(|&id| id as i32).collect()
                        } else {
                            ids.iter().map(|&id| id as i32).collect()
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        // Build Arrow ListArray
        let mut offsets = Vec::with_capacity(tokenized.len() + 1);
        let mut values = Vec::new();
        offsets.push(0i32);

        for ids in &tokenized {
            values.extend_from_slice(ids);
            offsets.push(values.len() as i32);
        }

        let values_array = Int32Array::from(values);
        let list_array = ListArray::try_new(
            Arc::new(Field::new("item", DataType::Int32, false)),
            arrow::buffer::OffsetBuffer::new(offsets.into()),
            Arc::new(values_array) as ArrayRef,
            None,
        )?;

        // Write output batch
        let output_batch = RecordBatch::try_new(
            output_schema.clone(),
            vec![Arc::new(list_array) as ArrayRef],
        )?;

        writer.write(&output_batch)?;

        total_processed += texts.len();
        pb.set_position(total_processed as u64);
    }

    pb.finish_with_message("Done!");
    writer.close()?;

    let elapsed = start_time.elapsed();
    let speed = total_processed as f64 / elapsed.as_secs_f64();

    println!("================================================================================");
    println!("✅ Tokenization completed!");
    println!("   Total samples: {}", total_processed);
    println!("   Time:          {:.1} min ({:.1} sec)", elapsed.as_secs_f64() / 60.0, elapsed.as_secs_f64());
    println!("   Speed:         {:.0} samples/sec", speed);
    println!("   Output:        {}", args.output.display());
    println!("================================================================================");

    Ok(())
}
