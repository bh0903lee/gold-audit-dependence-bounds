# SPHERE house recordings: source, license and checksums

- Data paper: Tonkin, E. L. et al. (2023). A multi-sensor dataset with annotated activities of daily living recorded in a residential setting. Scientific Data 10, 162. https://doi.org/10.1038/s41597-023-02017-1
- Dataset: University of Bristol data repository, doi:10.5523/bris.2h0wyctxrd69j2oqccsi45hy1p (version of 2022-11-04), https://data.bris.ac.uk/data/dataset/2h0wyctxrd69j2oqccsi45hy1p
- License: CC BY 4.0, as stated on the repository page. The README inside the dataset links to CC BY-SA 4.0.
- Included here: unmodified copies of the only files the analysis reads, from the dataset's `raw` view (`data/raw/NNN/` in the original archive, stored here as `raw/NNN/`): `meta.json`, `pir_raw.csv` and `location_K_raw.csv` for the 19 sessions (001-018, 021). They were downloaded one by one from `https://data.bris.ac.uk/datasets/2h0wyctxrd69j2oqccsi45hy1p/data/raw/NNN/FILE` on 2026-10-22. The full archive (586.6 MiB, with accelerometer, video features and dense tables) is not included.
- The checksums below equal `input_sha256` in `manuscript/supplement/extensions/sphere_summary.json`.

Recompute the values reported in Section 8.4 and Supplementary Table S6 (from the package root, about 30 s with 40 processes, about 10 min with one):

    python -I extensions/scripts/sphere_pir.py data/raw/sphere_house/raw results/generated/extensions/sphere 1000 40

With the pinned package versions (`requirements-cpu.lock`) and single-threaded BLAS, the output `summary.json` equals `manuscript/supplement/extensions/sphere_summary.json` byte for byte; other NumPy/SciPy versions reproduce the values up to floating-point rounding (relative differences below 1e-13), which leaves every value reported in Table S6 unchanged.

| File | Bytes | SHA-256 |
|---|---|---|
| raw/001/location_0_raw.csv | 522 | 0c655272b9ddf6b46abddfc5c32238cb209fcaac8f799c102e41da368dce758d |
| raw/001/location_1_raw.csv | 501 | 682e030d733fffa9b77746aed99923e8b4182796ea8e81c7a8c5acf85917a793 |
| raw/001/location_2_raw.csv | 429 | 6b451fdb02f991540649d3a2eb750dee7d871f630b310362092ae9ea0a5e7f98 |
| raw/001/meta.json | 90 | 38741bf4cbd1aace1f04735e7d7a0d7a8c3fd182706b4c4c4c6677b39615afbb |
| raw/001/pir_raw.csv | 4,825 | 564f748622a37156837b58db8bdecf007438f2c8334769c2284e3ffb290cfe05 |
| raw/002/location_0_raw.csv | 543 | ee3a100600489664ac6463fdc0b51800acfed433dd316b360b0042569a599a58 |
| raw/002/location_1_raw.csv | 464 | 72b1a3ba627c9ab6e798cfd79c2ab644c9104f68178c9476aa7ca51d44e3384b |
| raw/002/location_2_raw.csv | 530 | 03624dd4405df298fb8782ecdb91d48555e335405c597c6bc5dff3de987a41da |
| raw/002/meta.json | 90 | c403c0cd2feec4bf3ea6e528890033b4f44299110a7f3c9e3cf0a9f649bed4d1 |
| raw/002/pir_raw.csv | 2,911 | f08499a9ab9507f7d617383cc298f6dc60dfa53bd1559729b0e96f0e988f64de |
| raw/003/location_0_raw.csv | 457 | 866cad395d8789b0a3c34b504dd3e47f0d58dfc9c79ac79dc8391cc505fbefae |
| raw/003/location_1_raw.csv | 492 | ae4d42634878592c5d83852b815d8e75455b05c3a1ce56e5f76aa5deaae31295 |
| raw/003/location_2_raw.csv | 550 | 404325e7c078e07884300a6e24ab50438fdc59a230eec8e30d7b46340f4b07ab |
| raw/003/meta.json | 90 | 679b9721ee0f0817ec1898019aa0b92a4e8010a5fc111b67889e1464feab2a9a |
| raw/003/pir_raw.csv | 2,740 | 634cb7271dd57d3eeefee0b41d18ca6ed7f4918e5ccdc3885f13d20dc490e231 |
| raw/004/location_0_raw.csv | 555 | 242b1b2537d640710c2882ceba695a799b73c701d3dc091fc3d193b2010f821e |
| raw/004/location_1_raw.csv | 450 | fc1e69e3f8c9e4cd52e40b8d51293f6c4e2e4eae8fe31d889d3a8ef0b81613b8 |
| raw/004/location_2_raw.csv | 502 | dc8172cd16b439d795f9e0875d24e4595cde97fb7ac3c7400bb570eab1f3b537 |
| raw/004/meta.json | 91 | 64a1788c7f7a91f37c82d4384009814f3db974709722997b4040604b0e5c7f20 |
| raw/004/pir_raw.csv | 2,366 | eec2ccb7b4a11f6f550de3e9d050a11a9c89743174829457b4169448cfeedbaf |
| raw/005/location_0_raw.csv | 488 | 1148ff1bd7a4a8177d4b42375b990d169fb83e767856b676fc7283df17190f8d |
| raw/005/location_1_raw.csv | 431 | 7c82dbdf76578060ccb6cd0b5617713743979fb397a9b1ae2d8e9401343eaea0 |
| raw/005/meta.json | 83 | d758154875277aa8c904c01e82d7068530c2d9a3108d3e242dd1921813faddd9 |
| raw/005/pir_raw.csv | 3,748 | 08028460359722c5c446dc5fcc4efa4946d1dc254c0967449ab0f822719701ff |
| raw/006/location_0_raw.csv | 476 | 7119599016134fbcf8aceedc12fc38b2de06102521ec72b2e18e79bcc7b10b46 |
| raw/006/location_1_raw.csv | 425 | 7adb825c8e7cb58eef17ca9638b45ee6229c0c523fba733bde069da0d9b38648 |
| raw/006/meta.json | 84 | 7ea5ce0654dad659447c5838311e06eb38681053cf07d82944a741cd7347dc43 |
| raw/006/pir_raw.csv | 4,925 | 5716fb19cd146b07429c766d51b9c5b19aa082ed8ad89e40fa617cfce56186ab |
| raw/007/location_0_raw.csv | 383 | 421972e47b48b541e54ffcfc92086c86ad59527a0dfe40dd758326f87836c0e1 |
| raw/007/location_1_raw.csv | 572 | e0600366446afd406353a47cd9ed5049b668d54df1d3787d6a99dbd5518eb32c |
| raw/007/meta.json | 83 | ef2986b379b1b3636865ace88dc1a2cf90c164bef32383ce5e80424d457d928d |
| raw/007/pir_raw.csv | 3,671 | 04b5f3ed95065eb20fe4bbcbfe1f1a59acfe1bfab53139581249116eb17c8ce5 |
| raw/008/location_0_raw.csv | 538 | dae7e6837937fbcfb2f6ea65b5c21c1afcce6b35e6b724f3dfe04b9faa4465a4 |
| raw/008/location_1_raw.csv | 493 | f954ef875181733a5df1b0ae44075bf736288cace1c79de2b672d78848f5d657 |
| raw/008/meta.json | 83 | 5d7dc60a7380a030ed2425f32db978c051b8b9d8c1910b312c9b42a4c3f90c9e |
| raw/008/pir_raw.csv | 2,768 | a2459a91e596741eab536fdffecef55cfa175539d5d2902765c5cdab9f39eb72 |
| raw/009/location_0_raw.csv | 529 | 6cd4b559352bc37faf00f802d9d85313b33a6b9ebfae7cf4911594d112166b86 |
| raw/009/location_1_raw.csv | 582 | b75240fd66a19e3e9502ea8c5066e7668f77add111c861e1c011449529d80d80 |
| raw/009/location_2_raw.csv | 571 | deab4ec242ab15035586fd0357e17886bbad654686aed935e3e0b850a7d9a5fc |
| raw/009/meta.json | 90 | e9ccdde02c1f5683f536b381297738e6979d8042137b06152a74ab34db327938 |
| raw/009/pir_raw.csv | 2,821 | 448c4d6cf3fe6ac0724219786e054c139133ac1c138537bd95d9ff9e934ff9d8 |
| raw/010/location_0_raw.csv | 541 | 296737595a185bd44805981d42d78d8e0ee4f90bf69ed830369c870fde8dbae7 |
| raw/010/location_1_raw.csv | 442 | 0d92d8110f8468db356614ee03db7d49aa22af1cf437a5b3ca7190b974855f71 |
| raw/010/meta.json | 83 | 011e4f252728b55d282562f6dc6e4fc667c5885146af44343b3124982f0e751e |
| raw/010/pir_raw.csv | 2,413 | be148eb76da8bc49047bb34478a8349e3d7b214d5eca276c988973b7a41bdcad |
| raw/011/location_0_raw.csv | 561 | 78fd1518bc5e888543075c300a63aea67cebcb956db1d5eea8a6db74698541e9 |
| raw/011/location_1_raw.csv | 516 | af4f77aedf977228d9dbefbffc00f37a4546e4bf8b0e0c9298626cd8aa5b1ff0 |
| raw/011/meta.json | 83 | 61418ec7410451bc4ff5659a0831f62d2b31646bb74969564802d5197e6d822b |
| raw/011/pir_raw.csv | 2,642 | d2802331edc8b48977f725946b2cd78fd06a86a0ef4013c8515819783854f8b6 |
| raw/012/location_0_raw.csv | 460 | a20608304006d780834370a1199cf8ff88b83553f14ab5640656da7042959bb4 |
| raw/012/location_1_raw.csv | 488 | aebc469dd9b847b5fef5dd04be62b1dcf673e22f773555f09b3545a24eb2e9d7 |
| raw/012/meta.json | 85 | 273f2d614458c04726848f1ee4851bb0bafa650391adebfaea2d62699b900efa |
| raw/012/pir_raw.csv | 2,215 | c7f3a91c53ab044a2f84b2d792524e35fa4c74cc5d262e2fc3d1d617746b8971 |
| raw/013/location_0_raw.csv | 566 | 6c62f522a7f42dac58d6bb0e99d66db4539ba7e1329363477c63aa1d373b29e8 |
| raw/013/location_1_raw.csv | 451 | d021c7e757a4aaf10be3e613cb7a06683844bce5c0c51225487cbf7009852ba6 |
| raw/013/meta.json | 83 | 230d43a7fe5a9e5ce870a893c1a5db493f8f26677b10df159fe8731743f89e5a |
| raw/013/pir_raw.csv | 2,233 | 63abed47325b0c91ca5010d1d63d7ae0a5f7827a322bdf5c816543a420e403aa |
| raw/014/location_0_raw.csv | 563 | a527becbe7edc1b55e72538a232170b40317b7d5b883601fa4aadda6b328ef0b |
| raw/014/location_1_raw.csv | 423 | 591525507dc4e87ae55c7db0902e9f108c4c3618d396fead45a10cd108c607f0 |
| raw/014/meta.json | 83 | 12551c4b15cfc9d3fbe3f3b55273181283f7bfdd30375f46f8c6c2dfc6310949 |
| raw/014/pir_raw.csv | 2,810 | d0d53ee8279aff875a726fe93408039ac1c8b7b48103261f5738787dda8481af |
| raw/015/location_0_raw.csv | 435 | 0a98b4711db8a71b4c931b412d1f4d1729bdf97f9b13d516515850a12cdd3874 |
| raw/015/location_1_raw.csv | 545 | b33325c8e9f5180114604cf491bc19c72f9d6479361d6d699bc0ce8dc463d849 |
| raw/015/meta.json | 84 | f30e6e24b49e6c8be780564f1acc341cf18f8ddfff3041fbef90d99baebc7f41 |
| raw/015/pir_raw.csv | 3,804 | 70f8025c376de42c4b50a57a4acb2fa38e5f7a7510a0c828545b46b48ec50e03 |
| raw/016/location_0_raw.csv | 396 | fd49d7b3cdf705e46543a3e184c064d122437acb9b0b27fa7a00b448f874e8ff |
| raw/016/location_1_raw.csv | 424 | 39eb6eb953af826b8ae15c67050ded2f1f9fbf7da1b5b35b945729d329011e18 |
| raw/016/meta.json | 84 | 1aa8c6bf8ac949b9bf31d9184f633f8c2605ac0478e71b88ea07af3b89b240e9 |
| raw/016/pir_raw.csv | 2,559 | 4407679a682282df0243a1f8800cef8266cfc7bbd0135f56e5a5d3d9f2f6dba7 |
| raw/017/location_0_raw.csv | 494 | 4a3e1fa85c8cfe103f8316a91289ecb0ec3f2be812494ec60e4a16ea40274482 |
| raw/017/location_1_raw.csv | 582 | 8ded80290032dd0a02aeb31d8d82b0a1594bfc93c97d98929e9218bc40efe7e8 |
| raw/017/meta.json | 83 | 67b07c8648886903b23af3db6601877e48dbf04ee7435d407900323ae9171d69 |
| raw/017/pir_raw.csv | 2,673 | 050ffb0fcaeacf26bdba89688056b1692e35527ab663b5a421a80698e37021a8 |
| raw/018/location_0_raw.csv | 686 | cfdbf1c6184c7c281a97dfbcc41d29ed7ea24e4c73731739a4da1035bb94df6c |
| raw/018/location_1_raw.csv | 485 | 28f5de4a6fdacb1edcef0bb91ee0991ecf9ddd369e645a566f54f5f42ade9784 |
| raw/018/meta.json | 85 | 51639d0e87c9d3deb8c2d6bb1a15a97b0cb58b56e4a92af49031777b7f61c903 |
| raw/018/pir_raw.csv | 2,290 | 0fb595fbaccaae6801ba29130c23625bf39b64ad3d1ea72a06cb72a174b747a4 |
| raw/021/location_0_raw.csv | 611 | 821629da408fdd2ab648dc31b55246b041322f4dd2ceafc57812a1fbeb0d4e47 |
| raw/021/location_1_raw.csv | 481 | 41218e84ed8e2b08c7a2c97899296ca03812d64e351fe1fc610ce42e42a87c57 |
| raw/021/meta.json | 84 | 523e7361a66d6a04491bf69b2bdfa900dd55eae65610b142b29e88946229f75e |
| raw/021/pir_raw.csv | 3,952 | 4eeb5a53657b56c2b8c7ee9dc5e5d9c56dd2032ab04e61abb41812c723d52dae |
