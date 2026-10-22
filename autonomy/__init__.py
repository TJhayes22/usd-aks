"""Decision-making core for the AKS car: supervisor, planner, controller, safety.

Everything outside adapters/ is pure logic (stdlib + numpy) so it can be unit-tested
on a laptop and later wrapped as Donkey Car parts (each part class has a run() method).

Car frame everywhere: meters, x right, y forward, origin on the ground at the camera.
"""
