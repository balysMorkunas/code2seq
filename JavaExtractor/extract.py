#!/usr/bin/python

import itertools
import multiprocessing
import os
import shutil
import subprocess
import sys
from argparse import ArgumentParser
from threading import Timer


def get_immediate_subdirectories(a_dir):
    return [
        (os.path.join(a_dir, name))
        for name in os.listdir(a_dir)
        if os.path.isdir(os.path.join(a_dir, name))
    ]


TMP_DIR = ""


def ParallelExtractDir(args, dir):
    ExtractFeaturesForDir(args, dir, "")


def ExtractFeaturesForDir(args, dir, prefix):
    command = [
        "java",
        "-Xmx100g",
        "-XX:MaxNewSize=60g",
        "-cp",
        args.jar,
        "JavaExtractor.App",
        "--max_path_length",
        str(args.max_path_length),
        "--max_path_width",
        str(args.max_path_width),
        "--dir",
        dir,
        "--num_threads",
        str(args.num_threads),
    ]
    if args.inline_comments == "true":
        command.append("--inline_comments")

    # print command
    # os.system(command)
    kill = lambda process: process.kill()
    outputFileName = TMP_DIR + prefix + dir.split("/")[-1]
    failed = False
    with open(outputFileName, "a") as outputFile:
        sleeper = subprocess.Popen(command, stdout=outputFile, stderr=subprocess.PIPE)
        timer = Timer(60 * 60 * 60 * 60, kill, [sleeper])

        try:
            timer.start()
            stdout, stderr = sleeper.communicate()
        finally:
            timer.cancel()

        if sleeper.poll() == 0:
            if len(stderr) > 0:
                print(stderr, file=sys.stderr)
        else:
            print("dir: " + str(dir) + " was not completed in time", file=sys.stderr)
            failed = True
            subdirs = get_immediate_subdirectories(dir)
            for subdir in subdirs:
                ExtractFeaturesForDir(args, subdir, prefix + dir.split("/")[-1] + "_")
    if failed:
        if os.path.exists(outputFileName):
            os.remove(outputFileName)


def ExtractFeaturesForDirsList(args, dirs):
    global TMP_DIR
    TMP_DIR = "./tmp/feature_extractor%d/" % (os.getpid())
    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR, ignore_errors=True)
    os.makedirs(TMP_DIR)
    try:
        p = multiprocessing.Pool(6)
        p.starmap(ParallelExtractDir, zip(itertools.repeat(args), dirs))
        # for dir in dirs:
        #    ExtractFeaturesForDir(args, dir, '')
        output_files = os.listdir(TMP_DIR)
        for f in output_files:
            os.system("cat %s/%s" % (TMP_DIR, f))
    finally:
        shutil.rmtree(TMP_DIR, ignore_errors=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-maxlen",
        "--max_path_length",
        dest="max_path_length",
        required=False,
        default=8,
    )
    parser.add_argument(
        "-maxwidth",
        "--max_path_width",
        dest="max_path_width",
        required=False,
        default=2,
    )
    parser.add_argument(
        "-threads", "--num_threads", dest="num_threads", required=False, default=64
    )
    parser.add_argument("-j", "--jar", dest="jar", required=True)
    parser.add_argument("-dir", "--dir", dest="dir", required=False)
    parser.add_argument("-file", "--file", dest="file", required=False)
    parser.add_argument(
        "-inline_comments",
        "--inline_comments",
        dest="inline_comments",
        required=False,
        default=False,
    )
    args = parser.parse_args()

    if args.file is not None:
        command = (
            "java -cp "
            + args.jar
            + " JavaExtractor.App --max_path_length "
            + str(args.max_path_length)
            + " --max_path_width "
            + str(args.max_path_width)
            + " --file "
            + args.file
        )
        os.system(command)
    elif args.dir is not None:
        subdirs = get_immediate_subdirectories(args.dir)
        if len(subdirs) == 0:
            subdirs = [args.dir]
        ExtractFeaturesForDirsList(args, subdirs)
