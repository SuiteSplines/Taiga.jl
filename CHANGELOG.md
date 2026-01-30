## 0.6.0
 (2025-04-05)

### Features ✨

- [added linear elasticity module](m1ka05/taiga.jl@ce8a9e4b3b54fec2e73b19c481483fc35b4f1898) by @m1ka05

### Documentation 📚

- [updated link to linear elastic example](m1ka05/taiga.jl@1a8884c36697dc2df82ad6a903b440f9d96a2c99) by @m1ka05
- [updated docs module titles](m1ka05/taiga.jl@96a7c9ef08e69ca59b3ec12c3d0c8eebda3c1b03) by @m1ka05

## 0.5.0
 (2025-03-30)

### Features ✨

- [Added `PotentialFlowModule` and `ImmersedPotentialFlowModule`.](m1ka05/taiga.jl@c85e5197a7867df08827040777e020c7ec1d1f3d) by @m1ka05

### Continuous Integrations ⚙️

- [added html coverage report to pages job](m1ka05/taiga.jl@1f032bc640f22662a3893271d4f6ad6a5b9e04fd) by @m1ka05

## 0.4.0
 (2024-11-26)

### Code Refactoring 📦

- [refactored postprocessing and Bezier extraction](m1ka05/taiga.jl@73c48b6fe2846ae746aff772faef3fa63d29cb93) by @m1ka05

### Tests 🚨

- [fixed too strict niter tests (±1)](m1ka05/taiga.jl@25880db3ea3e05016b280577770d534a91b02d75) by @m1ka05

### Continuous Integrations ⚙️

- [updated documentation CI](m1ka05/taiga.jl@03f58f60db8196d0e57ef02a0d883fffb51cf212) by @m1ka05
- [updated documentation CI](m1ka05/taiga.jl@b00eeef967685ea1f0e1f761f8fc3aedb278e365) by @m1ka05
- [updated documentation CI](m1ka05/taiga.jl@9b7c04ce7de9006d5b5ddadc60a422b3440532ec) by @m1ka05
- [updated Docker image to Julia v1.10.6](m1ka05/taiga.jl@21b8cacb914d69ce92af30118d5de93742382c46) by @m1ka05

### Chores ♻️

- [updated details on versioning](m1ka05/taiga.jl@2ae359d6db1c052272fd5b04e21bbb8fd83b4e53) by @m1ka05

## 0.3.5
 (2024-10-29)

### Bug Fixes 🐛

- [Krylov.jl private API changed in v0.9.8](m1ka05/taiga.jl@5fabc7f28d7c23de97e62c3ece156e7bfa41c2d1) by @m1ka05

## 0.3.4
 (2024-03-13)

### Bug Fixes 🐛

- [set η̂ to η if posdef checks are skipped](m1ka05/taiga.jl@52931b695a41d5c6083ecc8bd43522378b55140e) by @m1ka05

## 0.3.3
 (2024-03-13)

### Features ✨

- [skipping of posdef checks in `InnerCG`/`InnerPCG`](m1ka05/taiga.jl@e730b732b116c9c02440588d6fcd6b16a9335492) by @m1ka05

## 0.3.2
 (2024-03-13)

### Features ✨

- [added restarts kwarg to extreme_eigenvalues()](m1ka05/taiga.jl@bf38e9993b098c8137e50af6c35cc6eab3149e2f) by @m1ka05

## 0.3.1
 (2024-03-13)

### Code Refactoring 📦

- [refactored convergence criterions](m1ka05/taiga.jl@eb445857a1f8802b00727f2869b7fc7b12e88f15) by @m1ka05

## v0.3.0
 (2024-03-08)

### Features ✨

- [added Poisson module](m1ka05/taiga.jl@4a05ffae9fe30a18dc29c615c115a0b8319dc012) by @m1ka05

### Tests 🚨

- [field tests use new Field constructors](m1ka05/taiga.jl@4134307aff9f675176459782c741a168429644df) by @m1ka05

### Continuous Integrations ⚙️

- [added KroneckerProducts to docs deps](m1ka05/taiga.jl@977163a2c9ea0ea9fe3ecc1ca81c7201301dad11) by @m1ka05

## v0.2.0
 (2024-02-09)

### Features ✨

- [added Field additional constructors](m1ka05/taiga.jl@189c64921c5e1c6a324dcfdfee662126f46c0486) by @m1ka05

### Documentation 📚

- [added documentation for `Domain` and `Partition`](m1ka05/taiga.jl@ef7e101cc090b3092923eb7cd94d76caa3987de5) by @m1ka05
- [added logo to documentation](m1ka05/taiga.jl@405de3d9c9bc0f40e413eee55a7e8796b394066c) by @m1ka05

### Tests 🚨

- [changed the way domains are defined in tests](m1ka05/taiga.jl@733a73ff82ef51060482d1ee73a0df1246e1f09d) by @m1ka05

### Continuous Integrations ⚙️

- [artifacts expiration set to 5 days](m1ka05/taiga.jl@eb34117e147d85c0a6a5c66f0b97784de181f871) by @m1ka05

### Reverts 🗑

- [Revert "artifacts will expire in 1 day"](m1ka05/taiga.jl@262383a37c81c698c2e079650c36b0a37950938f) by @m1ka05


## v0.1.4
 (2024-02-08)

### Continuous Integrations ⚙️

- [removed new line commit list](m1ka05/taiga.jl@d07aa2ca37c2a1991a6b3297ba5c5569a9105255) by @m1ka05


## v0.1.3
 (2024-02-08)

### Tests 🚨

- [add context to Bezier extraction with fields](m1ka05/taiga.jl@c3817a88d24d93adb35a1e5c08c910e35b4162ee) by @m1ka05

### Continuous Integrations ⚙️

- [updated changelog config](m1ka05/taiga.jl@fab0935299034567ee2aa4d87c348c1032332cd4) by @m1ka05
- [added author reference to changelog](m1ka05/taiga.jl@e204197df542e7f9448125e57b37c19369d16e1f) by @m1ka05


## v0.1.1
 (2024-02-08)

### Continuous Integrations (2 changes)

- [testing Gitlab changelog](m1ka05/taiga.jl@2e39d38690157b39d96c18e235223ae1ab71d8ab)
- [added configuration for Gitlab changelog](m1ka05/taiga.jl@28cd4e65aa055419e4f4f93615ef3465b684109b)
