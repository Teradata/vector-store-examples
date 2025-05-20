## Changelog

### Function Enhancements
- `set_auth_token` validates the authentication token using the `accounts` API provided by CCP team.
  - `create_context` can now authenticate to both database and CCP. 
    - Users can pass in the required parameters for `set_auth_token` in `create_context` function call itself.
    - Flexibility is provided to call `create_context` and `set_auth_token` separately as well. 
      - Note:
        - In case connection fails, appropriate error is raised and if the connection to CCP fails, 
          users can call `set_auth_token` instead of calling `create_context`.
          This will keep the database connection intact.