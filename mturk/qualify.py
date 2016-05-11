"""
Module for qualifying mturkers in bulk.
"""
import os, datetime, argparse, csv, logging.config
import requests
from . import utils

def assign_qualification(qualification_type_id, worker_id, integer_value,
        send_notification):
    """Assign qualification to a worker."""
    request_url = os.getenv("MTURK_URL")
    operation = "AssignQualification"
    timestamp = str(datetime.datetime.now().isoformat())
    request_params = {
            "Service": "AWSMechanicalTurkRequester",
            "AWSAccessKeyId": os.getenv("AWSACCESSKEYID"),
            "Operation": operation,
            "Signature": utils.generate_signature(operation, timestamp,
               os.getenv("AWSSECRETKEY")),
            "Timestamp": timestamp,
            "QualificationTypeId": qualification_type_id,
            "WorkerId": worker_id,
            "IntegerValue": integer_value,
            "SendNotification": send_notification
            };
    logging.debug(request_params)
    logging.info("Qualifying %s\t%s\t%s\t%r", qualification_type_id, worker_id,
            integer_value, send_notification)
    response = requests.get(request_url, params=request_params)
    logging.debug(response.content)
    logging.info("Qualifying %s\t%s\t%s\t%r", qualification_type_id,
            worker_id,integer_value, send_notification)


def qualify_workers(csvfile):
    """Assign qualification to all the workers in the file."""
    with open(csvfile, 'rb') as infile:
        data = csv.DictReader(infile) 
        for i, row in enumerate(data):
            print row
            assign_qualification(row["QualificationTypeId"], row["WorkerId"],
                row["IntegerValue"], row["SendNotification"])


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--qualify",
            help="Assign a qualification to the workers.",
            action='store_true')
    parser.add_argument("--input_file", help="File containing relevant info.")
    args = parser.parse_args()
    logging.config.fileConfig("logging_config.ini")
    if args.qualify:
        qualify_workers(args.input_file)


if __name__ == "__main__":
    main()

