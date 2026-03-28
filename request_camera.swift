import AVFoundation

let session = AVCaptureSession()
let discoverySession = AVCaptureDevice.DiscoverySession(deviceTypes: [.builtInWideAngleCamera], mediaType: .video, position: .unspecified)

if let device = discoverySession.devices.first {
    print("Camera found: \(device.localizedName)")
    do {
        let input = try AVCaptureDeviceInput(device: device)
        if session.canAddInput(input) {
            session.addInput(input)
            print("Camera access granted!")
        }
    } catch {
        print("Error accessing camera: \(error)")
    }
} else {
    print("No camera found")
}
