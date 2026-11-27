//! A pipeline library for XCL's own compute pipelines (`XCL_PSO_LIBRARY=1`, an experiment).
//!
//! Provenance: ported from Vice-Tay, https://github.com/TaynanGT/Vice-Tay, file
//! `crates/victory-gpu/src/pipeline/library.rs`, commit 1f69d67, GPL-3.0-or-later, ported on
//! 2026-11-27 and adapted to XCL's cache and pipeline paths. Copyright stays with its authors
//! (the file's commit in the shallow clone read for the port is by Vladimir; the full history
//! and `docs/credits.md` of Vice-Tay name the others). Ported: the mapped file with its handle
//! kept open, the save through a mapping and rename, the rejected-library start, one lock for
//! stores against a save, one at a time for loads of one name. Not ported: its graphics
//! pipelines, early list, page trim and device-removal notes. Listed in `THIRD_PARTY.md`.
//!
//! On some NVIDIA drivers a large compute pipeline made from `CachedPSO` takes as long as a
//! compile (10 – 30 s). An `ID3D12PipelineLibrary1` kept in a file is loaded in milliseconds
//! there (Vice-Tay measured about 20 ms each). A compute pipeline of `XCL_HUGE_PSO_BYTES` or
//! more is looked up here first (`real::compute_container`); one the library does not have is
//! made as before and stored. The library is saved under `<DXIL cache>\driver\` after
//! `SAVE_EVERY` new pipelines and when the process ends (`report::write_final`). The driver
//! refuses a library of another adapter or driver: an empty one is started in its place and
//! the file replaced.
//!
//! What went wrong there, kept here:
//! - The library's file is mapped, not read into memory: the driver keeps pointing into the
//!   bytes the library was made from for as long as it lives. The mapped file's handle stays
//!   open ([`Mapped`]): a save renames the new library over the file this run has mapped, and
//!   with the handle closed Windows gives up the old file's contents, so pages not in memory
//!   at that moment read as zeros later – and the driver, handed a pipeline of zeros, removes
//!   the device.
//! - The library is documented as safe to use from several threads, except that two loads of
//!   one name are made one after the other ([`same_name`]). One lock around every call made
//!   the pipeline workers wait for each other. A save takes the stores' lock alone: the size
//!   the library asks for must still be its size when it is written out.
//! - NVIDIA can restore a pipeline with static samplers from the process that stored it with
//!   the samplers in another order. A compute pipeline whose root signature has static
//!   samplers (`real::abi_rs_for`) is not kept here.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock, RwLock};

use windows::core::HSTRING;
use windows::Win32::Graphics::Direct3D12::*;

/// The library's file in the folder, and a saved library that could not take its place while
/// this run had it mapped (a system that refuses to replace a mapped file): the next start
/// renames it over the library first (`read_blob`).
const LIBRARY_FILE: &str = "compute-pipelines.bin";
const NEXT_FILE: &str = "compute-pipelines.next";
/// The library is saved after this many new pipelines.
const SAVE_EVERY: u64 = 16;

/// `XCL_PSO_LIBRARY=1`: huge compute pipelines go through the library (read once).
pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_PSO_LIBRARY").is_ok_and(|v| v.trim() == "1"))
}

/// A pipeline's name in the library: the name of its driver blob's file, so one program has one
/// entry.
pub(crate) fn entry_name(dxil: &[u8], flags: i32, chained: bool) -> String {
    super::real::driver_cache::name_for(dxil, flags, chained)
}

/// A whole file mapped into memory, read-only (the library's file) or writable (the file a save
/// fills). Unmapped when dropped.
struct Mapped {
    at: windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS,
    len: usize,
    /// The file of a mapping made by [`Mapped::open`]: open until after the view is unmapped, so
    /// that the file a rename took the name from keeps its contents for the view.
    _file: Option<std::fs::File>,
}

impl Mapped {
    /// Maps `file` (`len` bytes, not zero) for reading, or for reading and writing.
    fn map(file: &std::fs::File, len: usize, write: bool) -> Option<Mapped> {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MapViewOfFile, FILE_MAP_READ, FILE_MAP_WRITE, PAGE_READONLY, PAGE_READWRITE,
        };
        if len == 0 {
            return None;
        }
        // SAFETY: a file we hold open; the view keeps the file's pages after the mapping object
        // is closed.
        let at = unsafe {
            let mapping = CreateFileMappingW(
                file.as_raw_handle(),
                std::ptr::null(),
                if write { PAGE_READWRITE } else { PAGE_READONLY },
                0,
                0,
                std::ptr::null(),
            );
            if mapping.is_null() {
                return None;
            }
            let at = MapViewOfFile(mapping, if write { FILE_MAP_WRITE } else { FILE_MAP_READ }, 0, 0, 0);
            CloseHandle(mapping);
            at
        };
        (!at.Value.is_null()).then_some(Mapped { at, len, _file: None })
    }

    /// The file at `path`, mapped for reading with its handle kept; None if it is not there, is
    /// empty or cannot be mapped. Opened as the standard library opens a file to read: others may
    /// read, write, rename and delete it meanwhile, so a save still renames over it.
    fn open(path: &Path) -> Option<Mapped> {
        let file = std::fs::File::open(path).ok()?;
        let len = usize::try_from(file.metadata().ok()?.len()).ok()?;
        let mut mapped = Mapped::map(&file, len, false)?;
        mapped._file = Some(file);
        Some(mapped)
    }

    /// A new file of `len` bytes at `path` (replacing what is there), mapped for writing.
    fn create(path: &Path, len: usize) -> Option<Mapped> {
        let file = std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(true).open(path).ok()?;
        file.set_len(len as u64).ok()?;
        Mapped::map(&file, len, true)
    }

    fn bytes(&self) -> &[u8] {
        // SAFETY: the view holds `len` readable bytes until dropped.
        unsafe { std::slice::from_raw_parts(self.at.Value.cast(), self.len) }
    }

    /// Only for a view made by [`Mapped::create`].
    fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: a writable view of `len` bytes, ours alone.
        unsafe { std::slice::from_raw_parts_mut(self.at.Value.cast(), self.len) }
    }
}

impl Drop for Mapped {
    fn drop(&mut self) {
        // SAFETY: our view.
        unsafe { windows_sys::Win32::System::Memory::UnmapViewOfFile(self.at) };
    }
}

/// The bytes a library is made from: the driver reads them for as long as the library lives, so
/// they are kept beside it.
enum Blob {
    /// No file (an empty library).
    None,
    /// The file read into memory (it could not be mapped).
    Held(Vec<u8>),
    /// The file itself, mapped.
    Mapped(Mapped),
}

impl Blob {
    fn bytes(&self) -> &[u8] {
        match self {
            Blob::None => &[],
            Blob::Held(bytes) => bytes,
            Blob::Mapped(file) => file.bytes(),
        }
    }
}

/// The library file in `dir`: a library a save left beside it ([`NEXT_FILE`]) takes its place
/// first.
fn read_blob(dir: &Path) -> Blob {
    let path = dir.join(LIBRARY_FILE);
    let next = dir.join(NEXT_FILE);
    if next.exists() && std::fs::rename(&next, &path).is_err() {
        crate::xlog_warn!("D3D12.X: the pipeline library saved last could not take the old one's place");
    }
    if let Some(file) = Mapped::open(&path) {
        return Blob::Mapped(file);
    }
    match std::fs::read(&path) {
        Ok(bytes) if !bytes.is_empty() => Blob::Held(bytes),
        _ => Blob::None,
    }
}

/// The driver refused the library read from `dir` (another adapter or driver, damage): its bytes
/// are let go (a mapped file is unmapped) and the file is removed, so that the next save writes a
/// new one in its place.
fn discard_rejected(dir: &Path, blob: Blob) {
    drop(blob);
    let _ = std::fs::remove_file(dir.join(LIBRARY_FILE));
    let _ = std::fs::remove_file(dir.join(NEXT_FILE));
}

/// Writes a library of `size` bytes to a new file beside the library's file in `dir` (`fill`
/// writes it; through a mapping of that file, so that the bytes are the file's pages and not
/// the process's own memory for the seconds a save takes, through a buffer where the file cannot
/// be mapped) and renames it over the library. This run may have that file mapped: Windows 11
/// replaces a mapped file, and the mapping goes on reading the old one while a handle on it is
/// open ([`Mapped::open`]). Where a system refuses, the new library is left beside it as
/// [`NEXT_FILE`] for the next start. A failure leaves the old file and no temporary one.
fn write_library(dir: &Path, size: usize, fill: impl FnOnce(&mut [u8]) -> std::io::Result<()>) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let path = dir.join(LIBRARY_FILE);
    let tmp = path.with_extension(format!("tmp-{}", std::process::id()));
    let written = match Mapped::create(&tmp, size) {
        Some(mut file) => fill(file.bytes_mut()),
        None => {
            let mut bytes = vec![0u8; size];
            fill(&mut bytes).and_then(|()| std::fs::write(&tmp, &bytes))
        }
    };
    let done = written.and_then(|()| std::fs::rename(&tmp, &path).or_else(|_| std::fs::rename(&tmp, dir.join(NEXT_FILE))));
    if done.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    done
}

pub(crate) struct PipelineLibrary {
    library: ID3D12PipelineLibrary1,
    /// The blob the library was made from: it must outlive the library.
    _blob: Blob,
    dir: PathBuf,
    /// Pipelines stored since the last save.
    added: AtomicU64,
    /// Stores share it and a save takes it alone.
    saving: RwLock<()>,
}

// SAFETY: D3D12 objects are free-threaded, and the library synchronises its own calls (see the
// module's notes); what it does not, two loads of one name, is serialised by `same_name`.
unsafe impl Send for PipelineLibrary {}
// SAFETY: as above; the other fields are an atomic, a lock and a path.
unsafe impl Sync for PipelineLibrary {}

/// Held by a load of the pipeline named `name`: loads of the same name wait for each other (the
/// driver's one exception to the library's thread safety). Names share 64 locks.
fn same_name(name: &str) -> MutexGuard<'static, ()> {
    static LOADING: [Mutex<()>; 64] = [const { Mutex::new(()) }; 64];
    let at = name.bytes().fold(0usize, |h, b| h.wrapping_mul(31).wrapping_add(usize::from(b)));
    LOADING[at % LOADING.len()].lock().unwrap_or_else(|e| e.into_inner())
}

impl PipelineLibrary {
    /// The library kept in `dir`, or an empty one if there is none or the driver refuses it.
    /// `create` makes a library from bytes (none: an empty one) on the device.
    pub(crate) fn open(
        dir: &Path,
        create: impl Fn(Option<&[u8]>) -> windows::core::Result<ID3D12PipelineLibrary1>,
    ) -> Option<PipelineLibrary> {
        let blob = read_blob(dir);
        let (library, blob) = if blob.bytes().is_empty() {
            (create(None), Blob::None)
        } else {
            match create(Some(blob.bytes())) {
                Ok(library) => {
                    crate::xlog_info!(
                        "D3D12.X: pipeline library opened ({} KB, {})",
                        blob.bytes().len() >> 10,
                        if matches!(blob, Blob::Mapped(_)) { "mapped" } else { "read" }
                    );
                    (Ok(library), blob)
                }
                Err(e) => {
                    crate::xlog_info!("D3D12.X: pipeline library refused ({e}) – an empty one is started, its file replaced");
                    discard_rejected(dir, blob);
                    (create(None), Blob::None)
                }
            }
        };
        match library {
            Ok(library) => Some(PipelineLibrary {
                library,
                _blob: blob,
                dir: dir.to_path_buf(),
                added: AtomicU64::new(0),
                saving: RwLock::new(()),
            }),
            Err(e) => {
                crate::xlog_warn!("D3D12.X: no pipeline library ({e}): pipelines are not kept");
                None
            }
        }
    }

    /// The pipeline `name` made from `desc`, if the library has it with that description.
    pub(crate) fn load(&self, name: &str, desc: &D3D12_COMPUTE_PIPELINE_STATE_DESC) -> Option<ID3D12PipelineState> {
        let wide = HSTRING::from(name);
        let _one = same_name(name);
        // SAFETY: `desc` is a live description whose root signature outlives the call.
        unsafe { self.library.LoadComputePipeline::<_, ID3D12PipelineState>(&wide, desc) }.ok()
    }

    /// Keeps `pso` under `name` (a name already stored is refused, which leaves the library as
    /// it was; false); saves the library after `SAVE_EVERY` new ones.
    pub(crate) fn store(&self, name: &str, pso: &ID3D12PipelineState) -> bool {
        let wide = HSTRING::from(name);
        let stored = {
            let _stores = self.saving.read().unwrap_or_else(|e| e.into_inner());
            // SAFETY: a live pipeline.
            unsafe { self.library.StorePipeline(&wide, pso) }.is_ok()
        };
        if stored && self.added.fetch_add(1, Ordering::Relaxed) + 1 >= SAVE_EVERY {
            self.save();
        }
        stored
    }

    /// Writes the library if pipelines were stored since the last save. Does not wait: a store
    /// or a save of another thread in the way leaves it to the next call (the process may be
    /// ending with a thread cut off inside the library).
    pub(crate) fn save(&self) {
        static SAVING: Mutex<()> = Mutex::new(());
        let Ok(_one) = SAVING.try_lock() else { return };
        // Keeps stores out until the library is written: clearing `added` must not lose one.
        let Ok(_alone) = self.saving.try_write() else { return };
        if self.added.load(Ordering::Relaxed) == 0 {
            return;
        }
        let started = std::time::Instant::now();
        // SAFETY: a plain query.
        let size = unsafe { self.library.GetSerializedSize() };
        // SAFETY: a buffer of the size the library asks for.
        let result = write_library(&self.dir, size, |bytes| {
            unsafe { self.library.Serialize(bytes) }.map_err(|e| std::io::Error::other(e.to_string()))
        });
        match result {
            Ok(()) => {
                self.added.store(0, Ordering::Relaxed);
                crate::xlog_info!("D3D12.X: pipeline library saved ({} KB, {} ms)", size >> 10, started.elapsed().as_millis());
            }
            Err(e) => crate::xlog_warn!("D3D12.X: pipeline library not saved: {e}"),
        }
    }
}

static LIBRARY: OnceLock<Option<PipelineLibrary>> = OnceLock::new();

/// The process's library, opened on first use with `create` (see [`PipelineLibrary::open`]).
pub(crate) fn global(
    create: impl Fn(Option<&[u8]>) -> windows::core::Result<ID3D12PipelineLibrary1>,
) -> Option<&'static PipelineLibrary> {
    LIBRARY
        .get_or_init(|| PipelineLibrary::open(&crate::gpu::recomp::dxc::cache_dir().join("driver"), create))
        .as_ref()
}

/// The process ends (`report::write_final`): saves what was stored since the last save. Opens
/// nothing.
pub(crate) fn save_at_exit() {
    if let Some(Some(library)) = LIBRARY.get() {
        library.save();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::core::Interface;

    struct Scratch(PathBuf);
    impl Scratch {
        fn new(what: &str) -> Scratch {
            let dir = std::env::temp_dir().join(format!("xcl-pso-library-{what}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            Scratch(dir)
        }
    }
    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// One program has one entry, named as its driver blob's file.
    #[test]
    fn an_entry_is_named_like_the_blobs_file() {
        assert_eq!(entry_name(b"abc", 1, false), super::super::real::driver_cache::name(b"abc", 1));
        assert_ne!(entry_name(b"abc", 1, false), entry_name(b"abd", 1, false));
        assert_ne!(entry_name(b"abc", 1, false), entry_name(b"abc", 2, false));
        assert_ne!(entry_name(b"abc", 1, false), entry_name(b"abc", 1, true));
    }

    /// A refused library's file is gone (an empty library saved in its place), whether the
    /// driver read it from a mapping or from memory; no file reads as no library.
    #[test]
    fn a_refused_library_file_is_replaced() {
        let dir = Scratch::new("refused");
        let path = dir.0.join(LIBRARY_FILE);
        assert!(matches!(read_blob(&dir.0), Blob::None), "no file yet");
        std::fs::write(&path, b"").unwrap();
        assert!(matches!(read_blob(&dir.0), Blob::None), "an empty file");
        std::fs::write(&path, b"not a pipeline library".repeat(16)).unwrap();
        let blob = read_blob(&dir.0);
        assert!(matches!(blob, Blob::Mapped(_)));
        assert_eq!(blob.bytes(), &b"not a pipeline library".repeat(16)[..]);
        discard_rejected(&dir.0, blob);
        assert!(!path.exists());
        assert!(matches!(read_blob(&dir.0), Blob::None));
        // the same with a library left beside it
        std::fs::write(dir.0.join(NEXT_FILE), b"damaged").unwrap();
        discard_rejected(&dir.0, read_blob(&dir.0));
        assert!(!path.exists() && !dir.0.join(NEXT_FILE).exists());
    }

    /// A save writes a whole new file under a temporary name and renames it over the library
    /// while the old one is mapped (the mapping goes on reading the old bytes); a save that
    /// fails leaves the old file and no temporary one.
    #[test]
    fn a_save_replaces_the_file_atomically() {
        let dir = Scratch::new("save");
        let path = dir.0.join(LIBRARY_FILE);
        let old: Vec<u8> = (0..100_000u32).map(|i| i as u8).collect();
        write_library(&dir.0, old.len(), |b| Ok(b.copy_from_slice(&old))).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), old);
        let blob = read_blob(&dir.0);
        assert!(matches!(blob, Blob::Mapped(_)));
        let new: Vec<u8> = (0..150_000u32).map(|i| (i * 7) as u8).collect();
        write_library(&dir.0, new.len(), |b| Ok(b.copy_from_slice(&new))).unwrap();
        assert_eq!(blob.bytes(), &old[..], "the mapping still reads the library it was made from");
        drop(blob);
        assert_eq!(read_blob(&dir.0).bytes(), &new[..], "the next start reads the new one");
        assert_eq!(std::fs::read_dir(&dir.0).unwrap().count(), 1, "no temporary file left");
        let failed = write_library(&dir.0, 10, |_| Err(std::io::Error::other("serialize failed")));
        assert!(failed.is_err());
        assert_eq!(std::fs::read(&path).unwrap(), new, "the old library stays");
        assert_eq!(std::fs::read_dir(&dir.0).unwrap().count(), 1, "and no temporary file");
    }

    /// A pipeline made on the device is stored, the library saved, and a library opened again
    /// from the file loads it.
    #[test]
    fn a_stored_pipeline_loads_from_the_saved_file() {
        use crate::gpu::recomp;
        if let Err(reason) = recomp::dxc::location() {
            recomp::runner::skip_test(&format!("no DXC: {reason}"));
            return;
        }
        let Ok(dev) = recomp::runner::device() else {
            recomp::runner::skip_test("no D3D12 device");
            return;
        };
        let Ok(dev1) = dev.cast::<ID3D12Device1>() else {
            recomp::runner::skip_test("no ID3D12Device1");
            return;
        };
        let create = |bytes: Option<&[u8]>| unsafe {
            let (data, len) = bytes.map_or((std::ptr::null(), 0), |b| (b.as_ptr().cast(), b.len()));
            let mut out: Option<ID3D12PipelineLibrary1> = None;
            // (an empty slice is not a null blob: the empty library is made through the raw method)
            (Interface::vtable(&dev1).CreatePipelineLibrary)(
                Interface::as_raw(&dev1),
                data,
                len,
                &ID3D12PipelineLibrary1::IID,
                &mut out as *mut _ as *mut *mut std::ffi::c_void,
            )
            .ok()
            .map(|()| out.unwrap())
        };
        let dir = Scratch::new("gpu");
        let hlsl = "RWByteAddressBuffer mem : register(u2); [numthreads(1,1,1)] void main() { mem.Store(0, 42); }";
        let dxil = recomp::dxc::compile(hlsl, "main", "cs_6_6", &[]).expect("DXC compile");
        let rs = recomp::pso::abi_root_signature(&dev, false).expect("root signature");
        let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: unsafe { std::mem::transmute_copy(&rs) },
            CS: D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr().cast(), BytecodeLength: dxil.len() },
            ..Default::default()
        };
        let name = entry_name(&dxil, 0, false);
        let library = PipelineLibrary::open(&dir.0, &create).expect("an empty library");
        assert!(library.load(&name, &desc).is_none(), "nothing stored yet");
        let pso: ID3D12PipelineState = unsafe { dev.CreateComputePipelineState(&desc) }.expect("pipeline");
        assert!(library.store(&name, &pso));
        assert_eq!(library.added.load(Ordering::Relaxed), 1);
        library.save();
        assert_eq!(library.added.load(Ordering::Relaxed), 0);
        assert!(dir.0.join(LIBRARY_FILE).exists());
        let reopened = PipelineLibrary::open(&dir.0, &create).expect("the saved library");
        assert!(matches!(reopened._blob, Blob::Mapped(_)), "read from the file");
        assert!(reopened.load(&name, &desc).is_some(), "the stored pipeline loads");
        assert!(reopened.load("another-name", &desc).is_none());
    }
}
