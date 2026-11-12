module {
  func.func @mmad_lowering(
      %lhs: !tla.tensor<!tla.layout<!tla.shape<128,64>, !tla.stride<64,1>, !tla.shape<128,64>, zN>, !tla.coord<0,0>, !tla.ptr<f16, l0a, 512>>,
      %rhs: !tla.tensor<!tla.layout<!tla.shape<(16,4),(16,8)>, !tla.stride<(1,2048),(16,256)>, !tla.shape<64,128>, nZ>, !tla.coord<0,0>, !tla.ptr<f16, l0b, 512>>,
      %acc: !tla.tensor<!tla.layout<!tla.shape<128,128>, !tla.stride<128,1>, !tla.shape<128,128>, L0Clayout>, !tla.coord<0,0>, !tla.ptr<f32, l0c, 1024>>) {
    "tla.mmad"(%acc, %lhs, %rhs) {init_c = true}
        : (!tla.tensor<!tla.layout<!tla.shape<128,128>, !tla.stride<128,1>, !tla.shape<128,128>, L0Clayout>, !tla.coord<0,0>, !tla.ptr<f32, l0c, 1024>>,
           !tla.tensor<!tla.layout<!tla.shape<128,64>, !tla.stride<64,1>, !tla.shape<128,64>, zN>, !tla.coord<0,0>, !tla.ptr<f16, l0a, 512>>,
           !tla.tensor<!tla.layout<!tla.shape<(16,4),(16,8)>, !tla.stride<(1,2048),(16,256)>, !tla.shape<64,128>, nZ>, !tla.coord<0,0>, !tla.ptr<f16, l0b, 512>>) -> ()
    func.return
  }
}
