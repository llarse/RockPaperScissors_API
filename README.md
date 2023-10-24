# A Rock Paper Scissors AI API 

This project uses FAST api to build a rock paper scissors "light weight" DQN agent inference server to predict the users next move.

Allows the user to login but credentials are never "saved"
Instead, your JWT authentication is saved to load/save your A.I. checkpoint file. The A.I. utilized in this is a modified version of the RPS DQN agent found in my github

This is part of my "built on my day off" project series and was completed in one day so there are bound to be overlooked pieces.

Todo: - ADD PYTEST CODES BACK TO /TESTS
## Goals and Project Limitations
The main goals of this project are to demonstrate \
    1. Version Control (V1 utilizes a random response, V2 utilizes the DQN agent) \
    2. RFC 7519 JWT method authentication ( The implementation is semi-useless, but is done as a P.O.C. for utilizing JWT authentication, your username/password are not validated against a database and are just used to encode the JWT) \
    3. Production-ready API with empty docker configuration \
    4. Implementation of a "full bodied" (no segmented inference server) A.I. and wrapped agent in a API \
\
This project is NOT meant to demonstrate:\
    1. A production quality DQN inference server. The A.I. implemented in this project is done so without async calls, optimized instantiation to the GPU, checkpoint optimization, or user-specific instancing to save time during development as those were not the main focus. If you try to utilize the logic outlined in this project for a large-scale application, you would run into many efficiency "issues" and users would override the active agent if there is more than one user\
    2. User-specific instancing. As stated above, multiple users would override the active agent.\
    3. Full scale implementation of a JWT or a "typical" use case of one\
