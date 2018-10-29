from gym.envs.registration import register

register(
	id = 'DrivingOrigin-v0',
	entry_point = 'myenv.drivingOrigin:DrivingOriginEnv',
	)

register(
	id = 'PlanarQuad-v0',
	entry_point = 'myenv.planarQuad:PlanarQuadEnv',
	)
