## Changelog

### Vector Store Update Function Enhancements
The `update` function now accepts the following new parameters:

- **alter_operation**: Specifies the type of operation to perform for update (`ADD`/`DROP`).
- **update_style**: Specifies the style to be used with `alter_operation` (`MAJOR`/`MINOR`).
- **object_names**: Specifies the table name/`teradataml` DataFrame to be indexed for vector store.
- **ignore_embedding_errors**: Specifies whether to ignore errors during embedding generation.
- **database_name**: Specifies the database name where vector is created. User should have create table permission on this database.
- **batch**: Specifies whether to use batch processing for embedding generation. Applicable only for `aws`.

### Batch Processing Enhancements
- Batch support is enabled for `similarity_search`, `ask`, and `prepare_response` APIs.
- A new API `get_batch_result` is added to retrieve results after triggering the APIs in batch mode.

### Content-Based Vector Store Enhancements
- Content-based Vector Store can now be created on multiple tables/views.
- The `object_names` parameter can be provided with a list of strings or list of `DataFrame` while creating a Vector Store.

### Asynchronous Operations
- Creating, updating, and deleting Vector Stores is now asynchronous.
- The `status` API can be used to check the status of these operations.

### General Enhancements 
- Errors that were previously displayed in a DataFrame column are now thrown as `TeradatamlException`.
- DataFrame displayed to the user, except for `status()`, is now a teradataml DataFrame instead of a pandas DataFrame for better control.
- Implemented `get_details` function which gets the details of the Vector Store.

### Read/Write permissions are changed to User/Admin permissions.

### Bug Fixes
- Fixed disconnect issue to prevent "failed to disconnect" errors.
  - **Issue Reference**: *ELE-7574: [VS EARLY ACCESS] - teradataml: Inconsistent/erratic messaging behavior when disconnecting from a VS.*
- Fixed a bug where the `session_id` was not refreshed when the JWT was refreshed.
  - **Issue Reference**: *ELE-7624: [VS EARLY ACCESS] - Vector Store initialize or create fails after changing the authentication key and token.*
- `create` now displays an appropriate message if creation fails and is reinitialized.


