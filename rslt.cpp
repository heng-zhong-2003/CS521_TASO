namespace xla {
namespace {
absl::Status TASOVisitor::HandleAll(HloInstruction *root) {

  HloInstruction *str0, *str1, *str2;

          // Match source pattern
          if (Match(root, m::Add(m::Op(&str0), m::Op(&str1)))) {
        

  // Create replacement pattern
    HloInstruction* str3 = root->AddInstruction(HloInstruction::CreateBinary(root->shape(), HloOpcode::Add, str1, str0));
    return ReplaceWithNewInstruction(root, HloInstruction::CreateBinary(root->shape(), HloOpcode::Add, str1, str0));
  }

}
}
return absl::OkStatus();
}

