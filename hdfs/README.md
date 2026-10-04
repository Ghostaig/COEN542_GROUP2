# HDFS Configuration

This directory contains the Hadoop HDFS configuration used by the COEN542
network intrusion detection pipeline.

## Configuration

- `core-site.xml` sets the default filesystem to `hdfs://localhost:9000`.
- `hdfs-site.xml` configures:
  - HDFS replication factor: 1
  - NameNode metadata directory: `/mnt/bigdata/hdfs/namenode`
  - DataNode storage directory: `/mnt/bigdata/hdfs/datanode`

## Hadoop Version

The implementation was tested with Hadoop 3.4.1.

Hadoop uses Java 11 on the development machine:

```bash
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
```

Adjust this path if Java 11 is installed elsewhere.

## Start HDFS

```bash
start-dfs.sh
jps
hdfs dfsadmin -report
```

Expected services:

- NameNode
- DataNode
- SecondaryNameNode

## Initial Setup

Only on a fresh HDFS installation, format the NameNode once:

```bash
hdfs namenode -format
```

Do not run this command on an existing HDFS installation containing data.

Create the project directories:

```bash
hdfs dfs -mkdir -p /coen542/raw
hdfs dfs -mkdir -p /coen542/processed
hdfs dfs -mkdir -p /coen542/models
```

## Load Project Data

```bash
hdfs dfs -put data/streaming_store/network_flows /coen542/raw/
hdfs dfs -put data/processed/ml_binary /coen542/processed/
```

The processed ML dataset is:

```text
/coen542/processed/ml_binary
```

It contains 2,830,743 rows and 79 columns:
78 features plus `binary_label`.

## Verify

```bash
hdfs dfs -ls /coen542
hdfs dfs -ls /coen542/processed
hdfs dfsadmin -report
```

## Repository Note

The actual HDFS NameNode/DataNode storage directories are intentionally not
included in GitHub. They contain HDFS-managed internal data and are
machine-specific. The repository contains the configuration and instructions
needed to reproduce the HDFS setup.
