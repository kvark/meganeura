//! Structural keys: an op's variant and attributes as a token sequence,
//! with floats as their exact bits, so two ops are the same op exactly when
//! their keys are equal. Built on the ops' serde derive, which already
//! visits every attribute.

use serde::ser::{self, Serialize};

/// The key of any serializable value.
pub(crate) fn structural_key<T: Serialize + ?Sized>(value: &T) -> Vec<u64> {
    let mut keys = Keys(Vec::new());
    value
        .serialize(&mut keys)
        .expect("keys accept every serde value");
    keys.0
}

struct Keys(Vec<u64>);

#[derive(Debug)]
struct Never;

impl std::fmt::Display for Never {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("unreachable")
    }
}

impl std::error::Error for Never {}

impl ser::Error for Never {
    fn custom<T: std::fmt::Display>(_: T) -> Self {
        Never
    }
}

impl Keys {
    fn bytes(&mut self, v: &[u8]) {
        self.0.push(v.len() as u64);
        self.0.extend(v.iter().map(|&b| u64::from(b)));
    }
}

type Done = Result<(), Never>;

impl<'a> ser::Serializer for &'a mut Keys {
    type Ok = ();
    type Error = Never;
    type SerializeSeq = Self;
    type SerializeTuple = Self;
    type SerializeTupleStruct = Self;
    type SerializeTupleVariant = Self;
    type SerializeMap = Self;
    type SerializeStruct = Self;
    type SerializeStructVariant = Self;

    fn serialize_bool(self, v: bool) -> Done {
        self.0.push(u64::from(v));
        Ok(())
    }
    fn serialize_i8(self, v: i8) -> Done {
        self.serialize_i64(v.into())
    }
    fn serialize_i16(self, v: i16) -> Done {
        self.serialize_i64(v.into())
    }
    fn serialize_i32(self, v: i32) -> Done {
        self.serialize_i64(v.into())
    }
    fn serialize_i64(self, v: i64) -> Done {
        self.0.push(v as u64);
        Ok(())
    }
    fn serialize_u8(self, v: u8) -> Done {
        self.serialize_u64(v.into())
    }
    fn serialize_u16(self, v: u16) -> Done {
        self.serialize_u64(v.into())
    }
    fn serialize_u32(self, v: u32) -> Done {
        self.serialize_u64(v.into())
    }
    fn serialize_u64(self, v: u64) -> Done {
        self.0.push(v);
        Ok(())
    }
    fn serialize_f32(self, v: f32) -> Done {
        self.serialize_u64(v.to_bits().into())
    }
    fn serialize_f64(self, v: f64) -> Done {
        self.serialize_u64(v.to_bits())
    }
    fn serialize_char(self, v: char) -> Done {
        self.serialize_u64(u64::from(v))
    }
    fn serialize_str(self, v: &str) -> Done {
        self.bytes(v.as_bytes());
        Ok(())
    }
    fn serialize_bytes(self, v: &[u8]) -> Done {
        self.bytes(v);
        Ok(())
    }
    fn serialize_none(self) -> Done {
        self.serialize_u64(0)
    }
    fn serialize_some<T: Serialize + ?Sized>(self, value: &T) -> Done {
        self.0.push(1);
        value.serialize(self)
    }
    fn serialize_unit(self) -> Done {
        Ok(())
    }
    fn serialize_unit_struct(self, _: &'static str) -> Done {
        Ok(())
    }
    fn serialize_unit_variant(self, _: &'static str, index: u32, _: &'static str) -> Done {
        self.serialize_u32(index)
    }
    fn serialize_newtype_struct<T: Serialize + ?Sized>(self, _: &'static str, value: &T) -> Done {
        value.serialize(self)
    }
    fn serialize_newtype_variant<T: Serialize + ?Sized>(
        self,
        _: &'static str,
        index: u32,
        _: &'static str,
        value: &T,
    ) -> Done {
        self.0.push(index.into());
        value.serialize(self)
    }
    fn serialize_seq(self, len: Option<usize>) -> Result<Self, Never> {
        // Unknown lengths end with a marker no element token follows.
        self.0.push(len.map_or(u64::MAX, |n| n as u64));
        Ok(self)
    }
    fn serialize_tuple(self, _: usize) -> Result<Self, Never> {
        Ok(self)
    }
    fn serialize_tuple_struct(self, _: &'static str, _: usize) -> Result<Self, Never> {
        Ok(self)
    }
    fn serialize_tuple_variant(
        self,
        _: &'static str,
        index: u32,
        _: &'static str,
        _: usize,
    ) -> Result<Self, Never> {
        self.0.push(index.into());
        Ok(self)
    }
    fn serialize_map(self, len: Option<usize>) -> Result<Self, Never> {
        self.serialize_seq(len)
    }
    fn serialize_struct(self, _: &'static str, _: usize) -> Result<Self, Never> {
        Ok(self)
    }
    fn serialize_struct_variant(
        self,
        _: &'static str,
        index: u32,
        _: &'static str,
        _: usize,
    ) -> Result<Self, Never> {
        self.0.push(index.into());
        Ok(self)
    }
}

impl ser::SerializeSeq for &mut Keys {
    type Ok = ();
    type Error = Never;
    fn serialize_element<T: Serialize + ?Sized>(&mut self, value: &T) -> Done {
        value.serialize(&mut **self)
    }
    fn end(self) -> Done {
        Ok(())
    }
}

impl ser::SerializeTuple for &mut Keys {
    type Ok = ();
    type Error = Never;
    fn serialize_element<T: Serialize + ?Sized>(&mut self, value: &T) -> Done {
        value.serialize(&mut **self)
    }
    fn end(self) -> Done {
        Ok(())
    }
}

impl ser::SerializeTupleStruct for &mut Keys {
    type Ok = ();
    type Error = Never;
    fn serialize_field<T: Serialize + ?Sized>(&mut self, value: &T) -> Done {
        value.serialize(&mut **self)
    }
    fn end(self) -> Done {
        Ok(())
    }
}

impl ser::SerializeTupleVariant for &mut Keys {
    type Ok = ();
    type Error = Never;
    fn serialize_field<T: Serialize + ?Sized>(&mut self, value: &T) -> Done {
        value.serialize(&mut **self)
    }
    fn end(self) -> Done {
        Ok(())
    }
}

impl ser::SerializeMap for &mut Keys {
    type Ok = ();
    type Error = Never;
    fn serialize_key<T: Serialize + ?Sized>(&mut self, key: &T) -> Done {
        key.serialize(&mut **self)
    }
    fn serialize_value<T: Serialize + ?Sized>(&mut self, value: &T) -> Done {
        value.serialize(&mut **self)
    }
    fn end(self) -> Done {
        Ok(())
    }
}

impl ser::SerializeStruct for &mut Keys {
    type Ok = ();
    type Error = Never;
    fn serialize_field<T: Serialize + ?Sized>(&mut self, _: &'static str, value: &T) -> Done {
        value.serialize(&mut **self)
    }
    fn end(self) -> Done {
        Ok(())
    }
}

impl ser::SerializeStructVariant for &mut Keys {
    type Ok = ();
    type Error = Never;
    fn serialize_field<T: Serialize + ?Sized>(&mut self, _: &'static str, value: &T) -> Done {
        value.serialize(&mut **self)
    }
    fn end(self) -> Done {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::structural_key;
    use crate::graph::Op;

    #[test]
    fn floats_compare_by_bits() {
        let key = |value: f32| structural_key(&Op::Offset { value });
        assert_eq!(key(0.5), key(0.5));
        assert_ne!(key(0.0), key(-0.0));
        assert_ne!(key(f32::NAN), key(f32::from_bits(f32::NAN.to_bits() | 1)));
        assert_eq!(key(f32::NAN), key(f32::NAN));
        // Values Debug prints alike at low precision stay apart.
        assert_ne!(key(0.1), key(f32::from_bits(0.1f32.to_bits() + 1)));
    }

    #[test]
    fn variants_and_fields_are_distinguished() {
        assert_ne!(structural_key(&Op::Exp), structural_key(&Op::Log));
        assert_ne!(
            structural_key(&Op::RmsNorm { eps: 1e-5 }),
            structural_key(&Op::LayerNorm { eps: 1e-5 })
        );
        let shift = |offset| structural_key(&Op::ShiftInner { offset });
        assert_ne!(shift(1), shift(-1));
        assert_eq!(shift(3), shift(3));
        // Sequences carry their length, so a split point cannot move.
        let perm = |perm: Vec<usize>| structural_key(&Op::Permute { perm });
        assert_ne!(perm(vec![1, 0, 2]), perm(vec![1, 0]));
    }
}
