"""Data validation expectations for Crop Recommendation System."""
import sys
import logging
import great_expectations as ge

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_expectation_suite():
    """
    Create an expectation suite for crop recommendation dataset.

    Validates column existence, value ranges and null checks
    using Great Expectations framework.

    Returns:
        CheckpointResult: Result of validation checkpoint
    """
    try:
        context = ge.get_context()
        logger.info("Great Expectations context created")
    except Exception as e:
        logger.error("Failed to create GE context: %s", e)
        raise

    datasource = context.datasources["crop_data_source"]
    asset = datasource.assets[0]
    batch_requests = asset.build_batch_request()

    context.add_or_update_expectation_suite(
        "crop_recommendation_suite"
    )

    validator = context.get_validator(
        batch_request=batch_requests,
        expectation_suite_name="crop_recommendation_suite"
    )

    # Validate column existence
    columns = [
        "Nitrogen", "Phosphorus", "Potassium",
        "Temperature", "Humidity", "pH_Value",
        "Rainfall", "Crop"
    ]
    for col in columns:
        validator.expect_column_to_exist(col)
    logger.info("Column existence checks added")

    # Validate value ranges
    validator.expect_column_values_to_be_between(
        "Nitrogen", min_value=0, max_value=140
    )
    validator.expect_column_values_to_be_between(
        "Phosphorus", min_value=0, max_value=145
    )
    validator.expect_column_values_to_be_between(
        "Potassium", min_value=0, max_value=205
    )
    validator.expect_column_values_to_be_between(
        "Temperature", min_value=0, max_value=50
    )
    validator.expect_column_values_to_be_between(
        "Humidity", min_value=0, max_value=100
    )
    validator.expect_column_values_to_be_between(
        "pH_Value", min_value=0, max_value=14
    )
    validator.expect_column_values_to_be_between(
        "Rainfall", min_value=0, max_value=300
    )
    logger.info("Value range checks added")

    # Validate no null values
    for col in columns:
        validator.expect_column_values_to_not_be_null(col)
    logger.info("Null checks added")

    validator.save_expectation_suite(
        discard_failed_expectations=False
    )

    # Create and run checkpoint
    checkpoints = context.add_or_update_checkpoint(
        name="crop_recommendation_checkpoint",
        batch_request=batch_requests,
        expectation_suite_name="crop_recommendation_suite"
    )

    checkpoint_result = checkpoints.run()
    context.view_validation_result(checkpoint_result)

    if checkpoint_result.success:
        logger.info("Data validation passed successfully")
    else:
        logger.error(
            "Data validation failed ")
    data_docs = context.build_data_docs()
    logger.info("Data docs built: %s", data_docs)

    return checkpoint_result


if __name__ == "__main__":
    result = create_expectation_suite()
    sys.exit(0 if result.success else 1)
